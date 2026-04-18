use std::{collections::HashSet, path::PathBuf, sync::Arc};

use agent::{ThreadStore, ZED_AGENT_ID};
use agent_client_protocol as acp;
use anyhow::Result;
use chrono::{DateTime, Utc};
use collections::HashMap;
use db::{
    sqlez::{
        bindable::Column, domain::Domain, statement::Statement,
        thread_safe_connection::ThreadSafeConnection,
    },
    sqlez_macros::sql,
};
use feature_flags::{AgentV2FeatureFlag, FeatureFlagAppExt};
use gpui::{AppContext as _, Entity, Global, Subscription, Task};
use project::AgentId;
use serde::{Deserialize, Serialize};
use ui::{App, Context, SharedString};
use util::path_list::PathList;
use util::ResultExt;

pub fn init(cx: &mut App) {
    if ThreadMetadataStore::try_global(cx).is_none() {
        ThreadMetadataStore::init_global(cx);
    }

    if cx.has_flag::<AgentV2FeatureFlag>() {
        migrate_thread_metadata(cx);
    }
    cx.observe_flag::<AgentV2FeatureFlag, _>(|has_flag, cx| {
        if has_flag {
            migrate_thread_metadata(cx);
        }
    })
    .detach();
}

fn migrate_thread_metadata(cx: &mut App) {
    let store = ThreadMetadataStore::global(cx);
    let db = store.read(cx).db.clone();

    cx.spawn(async move |cx| {
        let existing_entries = db.list_ids()?.into_iter().collect::<HashSet<_>>();
        let is_first_migration = existing_entries.is_empty();

        let mut to_migrate = store.read_with(cx, |_store, cx| {
            ThreadStore::global(cx)
                .read(cx)
                .entries()
                .filter_map(|entry| {
                    if existing_entries.contains(&entry.id.0) {
                        return None;
                    }

                    Some(ThreadMetadata {
                        session_id: entry.id,
                        agent_id: ZED_AGENT_ID.clone(),
                        title: entry.title,
                        updated_at: entry.updated_at,
                        created_at: entry.created_at,
                        folder_paths: entry.folder_paths,
                        main_worktree_paths: PathList::default(),
                        archived: true,
                    })
                })
                .collect::<Vec<_>>()
        });

        if to_migrate.is_empty() {
            return anyhow::Ok(());
        }

        if is_first_migration {
            let mut per_project: HashMap<PathList, Vec<&mut ThreadMetadata>> = HashMap::default();
            for entry in &mut to_migrate {
                if entry.folder_paths.is_empty() {
                    continue;
                }
                per_project
                    .entry(entry.folder_paths.clone())
                    .or_default()
                    .push(entry);
            }
            for entries in per_project.values_mut() {
                entries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                for entry in entries.iter_mut().take(5) {
                    entry.archived = false;
                }
            }
        }

        for entry in to_migrate {
            db.save(entry).await?;
        }

        let _ = store.update(cx, |store, cx| store.reload(cx));
        anyhow::Ok(())
    })
    .detach_and_log_err(cx);
}

pub struct ThreadMetadataStore {
    db: ThreadMetadataDb,
    threads: HashMap<acp::SessionId, ThreadMetadata>,
    threads_by_paths: HashMap<Vec<PathBuf>, HashSet<acp::SessionId>>,
    #[allow(dead_code)]
    threads_by_main_paths: HashMap<Vec<PathBuf>, HashSet<acp::SessionId>>,
    #[allow(dead_code)]
    reload_task: Option<Task<Result<()>>>,
    #[allow(dead_code)]
    session_subscriptions: HashMap<acp::SessionId, Subscription>,
    pending_thread_ops_tx: smol::channel::Sender<Vec<DbOperation>>,
    _db_operations_task: Task<()>,
}

impl Global for ThreadMetadataStore {}

impl ThreadMetadataStore {
    pub fn try_global(cx: &App) -> Option<Entity<Self>> {
        cx.try_global::<GlobalThreadMetadataStore>()
            .map(|g| g.0.clone())
    }

    pub fn global(cx: &App) -> Entity<Self> {
        cx.global::<GlobalThreadMetadataStore>().0.clone()
    }

    pub fn init_global(cx: &mut App) {
        let store = cx.new(|cx| Self::new(cx));
        cx.set_global(GlobalThreadMetadataStore(store.clone()));
    }

    pub fn new(cx: &mut Context<Self>) -> Self {
        let db = THREAD_METADATA_DB.clone();

        let (tx, rx) = smol::channel::unbounded::<Vec<DbOperation>>();
        let _db_operations_task = cx.background_spawn({
            async move {
                while let Ok(first_update) = rx.recv().await {
                    let mut updates = vec![first_update];
                    while let Ok(update) = rx.try_recv() {
                        updates.push(update);
                    }
                    let updates = Self::dedup_db_operations(updates.into_iter().flatten().collect());
                    for operation in updates {
                        match operation {
                            DbOperation::Upsert(metadata) => {
                                THREAD_METADATA_DB.save(metadata).await.log_err();
                            }
                            DbOperation::Delete(session_id) => {
                                THREAD_METADATA_DB.delete(session_id).await.log_err();
                            }
                        }
                    }
                }
            }
        });

        let this = Self {
            db,
            threads: HashMap::default(),
            threads_by_paths: HashMap::default(),
            threads_by_main_paths: HashMap::default(),
            reload_task: None,
            session_subscriptions: HashMap::default(),
            pending_thread_ops_tx: tx,
            _db_operations_task,
        };
        let _ = this.reload(cx);
        this
    }

    pub fn entry(&self, session_id: &acp::SessionId) -> Option<&ThreadMetadata> {
        self.threads.get(session_id)
    }

    pub fn entries(&self) -> impl Iterator<Item = &ThreadMetadata> {
        self.threads.values()
    }

    pub fn entries_for_path(&self, path_list: &PathList) -> impl Iterator<Item = &ThreadMetadata> {
        let paths = path_list.paths();
        self.threads_by_paths
            .iter()
            .filter(move |(paths_key, _)| paths_key.as_slice() == paths)
            .flat_map(|(_, sessions)| sessions.iter())
            .filter_map(|session_id| self.threads.get(session_id))
            .filter(|thread| !thread.archived)
    }

    pub fn entries_for_main_worktree_path(
        &self,
        path_list: &PathList,
    ) -> impl Iterator<Item = &ThreadMetadata> {
        let paths = path_list.paths();
        self.threads_by_main_paths
            .iter()
            .filter(move |(paths_key, _)| paths_key.as_slice() == paths)
            .flat_map(|(_, sessions)| sessions.iter())
            .filter_map(|session_id| self.threads.get(session_id))
            .filter(|thread| !thread.archived)
    }

    pub fn archived_entries(&self) -> impl Iterator<Item = &ThreadMetadata> {
        self.threads.values().filter(|thread| thread.archived)
    }

    pub fn entry_ids(&self) -> impl Iterator<Item = &acp::SessionId> + '_ {
        self.threads.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    pub fn save(&mut self, metadata: ThreadMetadata, cx: &mut Context<Self>) {
        self.save_all(vec![metadata], cx);
    }

    pub fn save_all(&mut self, metadata: Vec<ThreadMetadata>, cx: &mut Context<Self>) {
        for metadata in &metadata {
            let was_archived = self
                .threads
                .get(&metadata.session_id)
                .map(|t| t.archived)
                .unwrap_or(false);
            self.threads.insert(metadata.session_id.clone(), metadata.clone());
            if metadata.archived && !was_archived {
                self.update_paths_index(&metadata);
            } else {
                self.remove_from_paths_index(&metadata.session_id);
                self.update_paths_index(&metadata);
            }
        }
        let ops = metadata
            .into_iter()
            .map(DbOperation::Upsert)
            .collect::<Vec<_>>();
        if let Err(err) = self.pending_thread_ops_tx.try_send(ops) {
            log::error!("failed to enqueue thread metadata save: {}", err);
        }
        cx.notify();
    }

    pub fn archive(&mut self, session_id: &acp::SessionId, cx: &mut Context<Self>) {
        if let Some(metadata_clone) = self.threads.get_mut(session_id).map(|metadata| {
            metadata.archived = true;
            metadata.clone()
        }) {
            self.update_paths_index(&metadata_clone);
            if let Err(err) = self.pending_thread_ops_tx.try_send(vec![DbOperation::Upsert(
                metadata_clone,
            )]) {
                log::error!("failed to enqueue thread metadata save: {}", err);
            }
            cx.notify();
        }
    }

    pub fn unarchive(&mut self, session_id: &acp::SessionId, cx: &mut Context<Self>) {
        if let Some(metadata_clone) = self.threads.get_mut(session_id).map(|metadata| {
            metadata.archived = false;
            metadata.clone()
        }) {
            self.update_paths_index(&metadata_clone);
            if let Err(err) = self.pending_thread_ops_tx.try_send(vec![DbOperation::Upsert(
                metadata_clone,
            )]) {
                log::error!("failed to enqueue thread metadata save: {}", err);
            }
            cx.notify();
        }
    }

    pub fn update_working_directories(
        &mut self,
        session_id: &acp::SessionId,
        folder_paths: PathList,
        cx: &mut Context<Self>,
    ) {
        let old_key = self
            .threads
            .get(session_id)
            .map(|metadata| metadata.folder_paths.paths().to_vec());
        if let Some(metadata) = self.threads.get_mut(session_id) {
            if let Some(old_key) = old_key {
                if let Some(paths) = self.threads_by_paths.get_mut(&old_key) {
                    paths.remove(session_id);
                }
            }
            metadata.folder_paths = folder_paths;
            let metadata_clone = metadata.clone();
            self.update_paths_index(&metadata_clone);
            if let Err(err) = self.pending_thread_ops_tx.try_send(vec![DbOperation::Upsert(
                metadata_clone,
            )]) {
                log::error!("failed to enqueue thread metadata save: {}", err);
            }
            cx.notify();
        }
    }

    pub fn delete(&mut self, session_id: acp::SessionId, cx: &mut Context<Self>) {
        self.remove_from_paths_index(&session_id);
        self.threads.remove(&session_id);
        if let Err(err) = self.pending_thread_ops_tx.try_send(vec![DbOperation::Delete(
            session_id.clone(),
        )]) {
            log::error!("failed to enqueue thread metadata delete: {}", err);
        }
        cx.notify();
    }

    pub fn reload(&self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let db = self.db.clone();
        cx.spawn(async move |this, cx| {
            let threads = db.list()?;
            let _ = this.update(cx, |this, cx| {
                let existing = std::mem::take(&mut this.threads);
                this.threads = threads.into_iter().map(|t| (t.session_id.clone(), t)).collect();
                let metadata_list: Vec<_> = this.threads.values().cloned().collect();
                for metadata in metadata_list {
                    this.update_paths_index(&metadata);
                }
                let removed: Vec<_> = existing
                    .keys()
                    .filter(|id| !this.threads.contains_key(id))
                    .cloned()
                    .collect();
                for session_id in removed {
                    this.remove_from_paths_index(&session_id);
                }
                cx.notify();
            });
            Ok(())
        })
    }

    fn update_paths_index(&mut self, metadata: &ThreadMetadata) {
        if metadata.archived {
            return;
        }
        let entry = metadata.session_id.clone();
        self.threads_by_paths
            .entry(metadata.folder_paths.paths().to_vec())
            .or_insert_with(HashSet::default)
            .insert(entry.clone());
        self.threads_by_main_paths
            .entry(metadata.main_worktree_paths.paths().to_vec())
            .or_insert_with(HashSet::default)
            .insert(entry);
    }

    fn remove_from_paths_index(&mut self, session_id: &acp::SessionId) {
        if let Some(metadata) = self.threads.get(session_id) {
            let entry = metadata.session_id.clone();
            let key = metadata.folder_paths.paths().to_vec();
            if let Some(paths) = self.threads_by_paths.get_mut(&key) {
                paths.remove(&entry);
            }
            let main_key = metadata.main_worktree_paths.paths().to_vec();
            if let Some(paths) = self.threads_by_main_paths.get_mut(&main_key) {
                paths.remove(&entry);
            }
        }
    }

    fn dedup_db_operations(operations: Vec<DbOperation>) -> Vec<DbOperation> {
        let mut ops = HashMap::default();
        for operation in operations.into_iter().rev() {
            if ops.contains_key(operation.id()) {
                continue;
            }
            ops.insert(operation.id().clone(), operation);
        }
        ops.into_values().collect()
    }
}

struct GlobalThreadMetadataStore(Entity<ThreadMetadataStore>);

impl Global for GlobalThreadMetadataStore {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMetadata {
    pub session_id: acp::SessionId,
    pub agent_id: AgentId,
    pub title: SharedString,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
    pub folder_paths: PathList,
    pub main_worktree_paths: PathList,
    pub archived: bool,
}

impl ThreadMetadata {
    pub fn new(
        session_id: acp::SessionId,
        agent_id: AgentId,
        title: SharedString,
        updated_at: DateTime<Utc>,
        folder_paths: PathList,
    ) -> Self {
        Self {
            session_id,
            agent_id,
            title,
            created_at: None,
            updated_at,
            folder_paths,
            main_worktree_paths: PathList::default(),
            archived: false,
        }
    }
}

#[derive(Clone)]
pub struct ThreadMetadataDb(ThreadSafeConnection);

impl Domain for ThreadMetadataDb {
    const NAME: &str = stringify!(ThreadMetadataDb);

    const MIGRATIONS: &[&str] = &[
        sql!(
            CREATE TABLE IF NOT EXISTS sidebar_threads(
                session_id TEXT PRIMARY KEY,
                agent_id TEXT,
                title TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_at TEXT,
                folder_paths TEXT,
                folder_paths_order TEXT
            ) STRICT;
        ),
        sql!(ALTER TABLE sidebar_threads ADD COLUMN archived INTEGER DEFAULT 0),
        sql!(ALTER TABLE sidebar_threads ADD COLUMN main_worktree_paths TEXT),
        sql!(ALTER TABLE sidebar_threads ADD COLUMN main_worktree_paths_order TEXT),
    ];
}

db::static_connection!(THREAD_METADATA_DB, ThreadMetadataDb, []);

impl ThreadMetadataDb {
    pub fn list_ids(&self) -> anyhow::Result<Vec<Arc<str>>> {
        self.select::<Arc<str>>(
            "SELECT session_id FROM sidebar_threads \
             ORDER BY updated_at DESC",
        )?()
    }

    pub fn list(&self) -> anyhow::Result<Vec<ThreadMetadata>> {
        self.select::<ThreadMetadata>(
            "SELECT session_id, agent_id, title, updated_at, created_at, folder_paths, folder_paths_order, archived, main_worktree_paths, main_worktree_paths_order \
             FROM sidebar_threads \
             ORDER BY updated_at DESC"
        )?()
    }

    pub async fn save(&self, row: ThreadMetadata) -> anyhow::Result<()> {
        let id = row.session_id.0.clone();
        let agent_id = if row.agent_id.as_ref() == ZED_AGENT_ID.as_ref() {
            None
        } else {
            Some(row.agent_id.to_string())
        };
        let title = row.title.to_string();
        let updated_at = row.updated_at.to_rfc3339();
        let created_at = row.created_at.map(|dt| dt.to_rfc3339());
        let serialized = row.folder_paths.serialize();
        let (folder_paths, folder_paths_order) = if row.folder_paths.is_empty() {
            (None, None)
        } else {
            (Some(serialized.paths), Some(serialized.order))
        };
        let main_serialized = row.main_worktree_paths.serialize();
        let (main_worktree_paths, main_worktree_paths_order) = if row.main_worktree_paths.is_empty()
        {
            (None, None)
        } else {
            (Some(main_serialized.paths), Some(main_serialized.order))
        };
        let archived = row.archived;

        self.write(move |conn| {
            let sql = "INSERT INTO sidebar_threads(session_id, agent_id, title, updated_at, created_at, folder_paths, folder_paths_order, archived, main_worktree_paths, main_worktree_paths_order) \
                       VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) \
                       ON CONFLICT(session_id) DO UPDATE SET \
                           agent_id = excluded.agent_id, \
                           title = excluded.title, \
                           updated_at = excluded.updated_at, \
                           created_at = excluded.created_at, \
                           folder_paths = excluded.folder_paths, \
                           folder_paths_order = excluded.folder_paths_order, \
                           archived = excluded.archived, \
                           main_worktree_paths = excluded.main_worktree_paths, \
                           main_worktree_paths_order = excluded.main_worktree_paths_order";
            let mut stmt = Statement::prepare(conn, sql)?;
            let mut i = stmt.bind(&id, 1)?;
            i = stmt.bind(&agent_id, i)?;
            i = stmt.bind(&title, i)?;
            i = stmt.bind(&updated_at, i)?;
            i = stmt.bind(&created_at, i)?;
            i = stmt.bind(&folder_paths, i)?;
            i = stmt.bind(&folder_paths_order, i)?;
            i = stmt.bind(&archived, i)?;
            i = stmt.bind(&main_worktree_paths, i)?;
            stmt.bind(&main_worktree_paths_order, i)?;
            stmt.exec()
        })
        .await
    }

    pub async fn delete(&self, session_id: acp::SessionId) -> anyhow::Result<()> {
        let id = session_id.0.clone();
        self.write(move |conn| {
            let mut stmt =
                Statement::prepare(conn, "DELETE FROM sidebar_threads WHERE session_id = ?")?;
            stmt.bind(&id, 1)?;
            stmt.exec()
        })
        .await
    }
}

impl Column for ThreadMetadata {
    fn column(statement: &mut Statement, start_index: i32) -> anyhow::Result<(Self, i32)> {
        let (id, next): (Arc<str>, i32) = Column::column(statement, start_index)?;
        let (agent_id, next): (Option<String>, i32) = Column::column(statement, next)?;
        let (title, next): (String, i32) = Column::column(statement, next)?;
        let (updated_at_str, next): (String, i32) = Column::column(statement, next)?;
        let (created_at_str, next): (Option<String>, i32) = Column::column(statement, next)?;
        let (folder_paths_str, next): (Option<String>, i32) = Column::column(statement, next)?;
        let (folder_paths_order_str, next): (Option<String>, i32) =
            Column::column(statement, next)?;
        let (archived, next): (bool, i32) = Column::column(statement, next)?;
        let (main_worktree_paths_str, next): (Option<String>, i32) =
            Column::column(statement, next)?;
        let (main_worktree_paths_order_str, next): (Option<String>, i32) =
            Column::column(statement, next)?;

        let agent_id = agent_id
            .map(|id| AgentId::new(id))
            .unwrap_or_else(|| ZED_AGENT_ID.clone());

        let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)?.with_timezone(&Utc);
        let created_at = created_at_str
            .as_deref()
            .map(DateTime::parse_from_rfc3339)
            .transpose()?
            .map(|dt| dt.with_timezone(&Utc));

        let folder_paths = folder_paths_str
            .map(|paths| {
                PathList::deserialize(&util::path_list::SerializedPathList {
                    paths,
                    order: folder_paths_order_str.unwrap_or_default(),
                })
            })
            .unwrap_or_default();

        let main_worktree_paths = main_worktree_paths_str
            .map(|paths| {
                PathList::deserialize(&util::path_list::SerializedPathList {
                    paths,
                    order: main_worktree_paths_order_str.unwrap_or_default(),
                })
            })
            .unwrap_or_default();

        Ok((
            ThreadMetadata {
                session_id: acp::SessionId::new(id),
                agent_id,
                title: title.into(),
                updated_at,
                created_at,
                folder_paths,
                main_worktree_paths,
                archived,
            },
            next,
        ))
    }
}

pub enum DbOperation {
    Upsert(ThreadMetadata),
    Delete(acp::SessionId),
}

impl DbOperation {
    fn id(&self) -> &acp::SessionId {
        match self {
            DbOperation::Upsert(metadata) => &metadata.session_id,
            DbOperation::Delete(session_id) => session_id,
        }
    }
}

impl PartialEq for DbOperation {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for DbOperation {}

impl std::hash::Hash for DbOperation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
