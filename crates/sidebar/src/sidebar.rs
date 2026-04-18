mod thread_switcher;

use acp_thread::{AgentSessionInfo, ThreadStatus};
use agent_client_protocol as acp;
use agent_ui::{
    AgentPanel, AgentPanelEvent, NewThread, ThreadMetadata, ThreadMetadataStore,
    ThreadsArchiveView, ThreadsArchiveViewEvent,
};
use db::kvp::KEY_VALUE_STORE;
use fs::Fs;
use fuzzy::StringMatchCandidate;
use gpui::{
    Action as _, App, Context, Entity, EventEmitter, FocusHandle, Focusable, Pixels, Render,
    SharedString, Subscription, Task, Window, px,
};
use picker::{Picker, PickerDelegate};
use project::ProjectGroupKey;
use project::agent_server_store::{CLAUDE_CODE_NAME, CODEX_NAME, GEMINI_NAME};
use project::git_store::linked_worktree_short_name;
use project::Event as ProjectEvent;
use recent_projects::{RecentProjectEntry, get_recent_projects};

use std::collections::{HashMap, HashSet};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use theme::ActiveTheme;
use time::OffsetDateTime;
use ui::utils::TRAFFIC_LIGHT_PADDING;
use ui::{
    CommonAnimationExt, Disclosure, Divider, DividerColor, KeyBinding, ListSubHeader, Tab,
    ThreadItem, Tooltip, prelude::*,
};
use ui_input::ErasedEditor;
use util::ResultExt as _;
use workspace::{
    FocusWorkspaceSidebar, MultiWorkspace, NewWorkspaceInWindow, OpenMode,
    Sidebar as WorkspaceSidebar, SidebarEvent, ToggleWorkspaceSidebar, Workspace,
};
use crate::thread_switcher::{ThreadSwitcher, ThreadSwitcherEntry, ThreadSwitcherEvent};

gpui::actions!(
    agents_sidebar,
    [NewThreadInGroup, ToggleArchive, ToggleThreadSwitcher, FocusSidebarFilter]
);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentThreadStatus {
    Running,
    Completed,
}

#[derive(Clone, Debug)]
struct AgentThreadInfo {
    title: SharedString,
    status: AgentThreadStatus,
    generating_title: bool,
}

#[derive(Clone)]
struct ProjectThreadSummary {
    metadata: ThreadMetadata,
    status: AgentThreadStatus,
    generating_title: bool,
    is_active: bool,
    icon: IconName,
    worktree_label: SharedString,
    full_path: SharedString,
}

const LAST_THREAD_TITLES_KEY: &str = "sidebar-last-thread-titles";

const DEFAULT_WIDTH: Pixels = px(320.0);
const MIN_WIDTH: Pixels = px(200.0);
const MAX_WIDTH: Pixels = px(800.0);
const MAX_MATCHES: usize = 100;
const DEFAULT_THREADS_SHOWN: usize = 5;

#[derive(Clone)]
struct ProjectGroupEntry {
    key: ProjectGroupKey,
    activation_index: usize,
    workspace_indices: Vec<usize>,
    worktree_label: SharedString,
    threads: Vec<ProjectThreadSummary>,
    thread_info: Option<AgentThreadInfo>,
    draft_text: Option<SharedString>,
    has_running_threads: bool,
    is_remote: bool,
}

impl ProjectGroupEntry {
    fn thread_icon(agent_id: &str) -> IconName {
        match agent_id {
            "Zed Agent" => IconName::ZedAgent,
            GEMINI_NAME => IconName::AiGemini,
            CLAUDE_CODE_NAME => IconName::AiClaude,
            CODEX_NAME => IconName::Terminal,
            _ => IconName::Terminal,
        }
    }

    fn thread_worktree_label(key: &ProjectGroupKey, metadata: &ThreadMetadata) -> SharedString {
        let main_paths = key.path_list().paths();
        let mut names = Vec::new();

        for path in metadata.folder_paths.paths() {
            if main_paths.iter().any(|main_path| main_path.as_path() == path.as_path()) {
                continue;
            }

            let label = main_paths
                .iter()
                .find_map(|main_path| linked_worktree_short_name(main_path, path))
                .unwrap_or_else(|| {
                    path.file_name()
                        .map(|name| name.to_string_lossy().to_string().into())
                        .unwrap_or_else(|| path.to_string_lossy().to_string().into())
                });
            names.push(label.to_string());
        }

        if names.is_empty() {
            key.display_name()
        } else {
            names.join(", ").into()
        }
    }

    fn thread_full_path(metadata: &ThreadMetadata) -> SharedString {
        metadata
            .folder_paths
            .paths()
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("\n")
            .into()
    }

    fn thread_count(&self) -> usize {
        self.threads.len() + usize::from(self.draft_text.is_some())
    }

    fn new(
        key: ProjectGroupKey,
        workspaces: &[(usize, Entity<Workspace>)],
        active_workspace_index: usize,
        persisted_titles: &HashMap<String, String>,
        cx: &App,
    ) -> Self {
        let workspace_indices = workspaces.iter().map(|(index, _)| *index).collect::<Vec<_>>();
        let activation_index = workspaces
            .iter()
            .find(|(index, _)| *index == active_workspace_index)
            .map(|(index, _)| *index)
            .unwrap_or_else(|| workspaces[0].0);

        let worktree_label = key.display_name();
        let active_session_id = workspaces
            .iter()
            .find(|(index, _)| *index == active_workspace_index)
            .and_then(|(_, workspace)| Self::active_session_id(workspace, cx));
        let store = ThreadMetadataStore::global(cx);
        let mut seen_session_ids = HashSet::new();
        let mut thread_metadata = Vec::new();
        {
            let store = store.read(cx);
            for metadata in store.entries_for_main_worktree_path(key.path_list()) {
                if seen_session_ids.insert(metadata.session_id.clone()) {
                    thread_metadata.push(metadata.clone());
                }
            }
            for metadata in store.entries_for_path(key.path_list()) {
                if seen_session_ids.insert(metadata.session_id.clone()) {
                    thread_metadata.push(metadata.clone());
                }
            }
        }

        let mut threads: Vec<ProjectThreadSummary> = thread_metadata
            .into_iter()
            .map(|metadata| {
                let live_info = Self::thread_info_for_metadata(workspaces, &metadata, cx);
                let (status, generating_title, title) = if let Some(info) = live_info {
                    (info.status, info.generating_title, info.title)
                } else {
                    (
                        AgentThreadStatus::Completed,
                        false,
                        metadata.title.clone(),
                    )
                };

                let mut metadata = metadata;
                metadata.title = title;
                ProjectThreadSummary {
                    is_active: active_session_id
                        .as_ref()
                        .is_some_and(|session_id| session_id == metadata.session_id.0.as_ref()),
                    icon: Self::thread_icon(metadata.agent_id.as_ref()),
                    worktree_label: Self::thread_worktree_label(&key, &metadata),
                    full_path: Self::thread_full_path(&metadata),
                    metadata,
                    status,
                    generating_title,
                }
            })
            .collect();
        threads.sort_by(|a, b| b.metadata.updated_at.cmp(&a.metadata.updated_at));

        let has_running_threads = threads
            .iter()
            .any(|thread| thread.status == AgentThreadStatus::Running);

        let thread_info = workspaces
            .iter()
            .find(|(index, _)| *index == active_workspace_index)
            .and_then(|(_, workspace)| Self::thread_info(workspace, cx))
            .or_else(|| {
                workspaces
                    .iter()
                    .find_map(|(_, workspace)| Self::thread_info(workspace, cx))
            })
            .or_else(|| {
            threads.first().map(|thread| AgentThreadInfo {
                title: thread.metadata.title.clone(),
                status: thread.status.clone(),
                generating_title: thread.generating_title,
            })
            })
            .or_else(|| {
            if key.path_list().paths().is_empty() {
                return None;
            }
            let path_key = sorted_paths_key(key.path_list().paths());
            let title = persisted_titles.get(&path_key)?;
            Some(AgentThreadInfo {
                title: SharedString::from(title.clone()),
                status: AgentThreadStatus::Completed,
                generating_title: false,
            })
        });

        let draft_text = workspaces
            .iter()
            .find(|(index, _)| *index == active_workspace_index)
            .and_then(|(_, workspace)| {
                let agent_panel = workspace.read(cx).panel::<AgentPanel>(cx)?;
                agent_panel.read(cx).active_agent_draft_text(cx)
            });

        let is_remote = key.host().is_some();

        Self {
            key,
            activation_index,
            workspace_indices,
            worktree_label,
            threads,
            thread_info,
            draft_text,
            has_running_threads,
            is_remote,
        }
    }

    fn thread_info(workspace: &Entity<Workspace>, cx: &App) -> Option<AgentThreadInfo> {
        let agent_panel = workspace.read(cx).panel::<AgentPanel>(cx)?;
        let thread = agent_panel.read(cx).active_agent_thread(cx)?;
        let thread_ref = thread.read(cx);
        let title = thread_ref.title();
        let status = match thread_ref.status() {
            ThreadStatus::Generating => AgentThreadStatus::Running,
            ThreadStatus::Idle => AgentThreadStatus::Completed,
        };
        let generating_title = status == AgentThreadStatus::Running && title.is_empty();
        Some(AgentThreadInfo {
            title,
            status,
            generating_title,
        })
    }

    fn active_session_id(workspace: &Entity<Workspace>, cx: &App) -> Option<String> {
        let agent_panel = workspace.read(cx).panel::<AgentPanel>(cx)?;
        let thread = agent_panel.read(cx).active_agent_thread(cx)?;
        Some(thread.read(cx).session_id().0.to_string())
    }

    fn thread_info_for_metadata(
        workspaces: &[(usize, Entity<Workspace>)],
        metadata: &ThreadMetadata,
        cx: &App,
    ) -> Option<AgentThreadInfo> {
        workspaces.iter().find_map(|(_, workspace)| {
            let agent_panel = workspace.read(cx).panel::<AgentPanel>(cx)?;
            let thread = agent_panel.read(cx).active_agent_thread(cx)?;
            let thread_ref = thread.read(cx);
            if thread_ref.session_id() != &metadata.session_id {
                return None;
            }

            let title = thread_ref.title();
            let status = match thread_ref.status() {
                ThreadStatus::Generating => AgentThreadStatus::Running,
                ThreadStatus::Idle => AgentThreadStatus::Completed,
            };
            let generating_title = status == AgentThreadStatus::Running && title.is_empty();
            Some(AgentThreadInfo {
                title,
                status,
                generating_title,
            })
        })
    }
}

#[derive(Clone)]
enum SidebarEntry {
    Separator(SharedString),
    ProjectHeader(ProjectGroupEntry),
    ProjectDraftThread {
        group: ProjectGroupEntry,
        title: SharedString,
    },
    ProjectThread {
        group: ProjectGroupEntry,
        thread: ProjectThreadSummary,
    },
    ProjectViewMore {
        group: ProjectGroupEntry,
        shown: usize,
        total: usize,
        is_fully_expanded: bool,
    },
    ProjectNewThread(ProjectGroupEntry),
    RecentProject(RecentProjectEntry),
}

impl SidebarEntry {
    fn searchable_text(&self) -> &str {
        match self {
            SidebarEntry::Separator(_) => "",
            SidebarEntry::ProjectHeader(entry) => entry.worktree_label.as_ref(),
            SidebarEntry::ProjectDraftThread { title, .. } => title.as_ref(),
            SidebarEntry::ProjectThread { thread, .. } => thread.metadata.title.as_ref(),
            SidebarEntry::ProjectViewMore { .. } => "",
            SidebarEntry::ProjectNewThread(_) => "New Thread",
            SidebarEntry::RecentProject(entry) => entry.name.as_ref(),
        }
    }
}

#[derive(Clone)]
struct SidebarMatch {
    entry: SidebarEntry,
    positions: Vec<usize>,
}

enum SidebarView {
    ThreadList,
    Archive(Entity<ThreadsArchiveView>),
}

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
enum SerializedSidebarView {
    #[default]
    ThreadList,
    Archive,
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct SerializedSidebarState {
    #[serde(default)]
    width: Option<f32>,
    #[serde(default)]
    active_view: SerializedSidebarView,
    #[serde(default)]
    collapsed_groups: Vec<Vec<PathBuf>>,
    #[serde(default)]
    expanded_groups: Vec<(Vec<PathBuf>, usize)>,
}

struct WorkspacePickerDelegate {
    multi_workspace: Entity<MultiWorkspace>,
    entries: Vec<SidebarEntry>,
    active_group_key: Option<ProjectGroupKey>,
    project_group_count: usize,
    /// All recent projects including what's filtered out of entries
    /// used to add unopened projects to entries on rebuild
    recent_projects: Vec<RecentProjectEntry>,
    recent_project_thread_titles: HashMap<SharedString, SharedString>,
    matches: Vec<SidebarMatch>,
    selected_index: usize,
    query: String,
    hovered_thread_item: Option<ProjectGroupKey>,
    notified_groups: HashSet<ProjectGroupKey>,
    collapsed_groups: HashSet<ProjectGroupKey>,
    expanded_groups: HashMap<ProjectGroupKey, usize>,
}

impl WorkspacePickerDelegate {
    fn new(multi_workspace: Entity<MultiWorkspace>) -> Self {
        Self {
            multi_workspace,
            entries: Vec::new(),
            active_group_key: None,
            project_group_count: 0,
            recent_projects: Vec::new(),
            recent_project_thread_titles: HashMap::new(),
            matches: Vec::new(),
            selected_index: 0,
            query: String::new(),
            hovered_thread_item: None,
            notified_groups: HashSet::new(),
            collapsed_groups: HashSet::new(),
            expanded_groups: HashMap::new(),
        }
    }

    fn set_entries(
        &mut self,
        project_groups: Vec<ProjectGroupEntry>,
        active_group_key: Option<ProjectGroupKey>,
        cx: &App,
    ) {
        let project_group_keys: HashSet<ProjectGroupKey> =
            project_groups.iter().map(|group| group.key.clone()).collect();
        if let Some(hovered_key) = self.hovered_thread_item.as_ref() {
            let still_exists = project_groups
                .iter()
                .any(|group| &group.key == hovered_key);
            if !still_exists {
                self.hovered_thread_item = None;
            }
        }
        self.expanded_groups
            .retain(|group_key, _| project_group_keys.contains(group_key));
        self.collapsed_groups
            .retain(|group_key| project_group_keys.contains(group_key));

        let old_statuses: HashMap<ProjectGroupKey, AgentThreadStatus> = self
            .entries
            .iter()
            .filter_map(|entry| match entry {
                SidebarEntry::ProjectThread { group, thread } => {
                    Some((group.key.clone(), thread.status.clone()))
                }
                _ => None,
            })
            .collect();

        for thread in &project_groups {
            if let Some(info) = &thread.thread_info {
                if info.status == AgentThreadStatus::Completed {
                    let is_active_group = active_group_key
                        .as_ref()
                        .is_some_and(|key| key == &thread.key);
                    if !is_active_group
                        && old_statuses.get(&thread.key) == Some(&AgentThreadStatus::Running)
                    {
                        self.notified_groups.insert(thread.key.clone());
                    }
                }
            }
        }

        if self.active_group_key != active_group_key {
            if let Some(previous_key) = &self.active_group_key {
                self.notified_groups.remove(previous_key);
            }
        }
        self.active_group_key = active_group_key;
        self.project_group_count = project_groups.len();
        self.rebuild_entries(project_groups, cx);
    }

    fn current_project_groups(&self) -> Vec<ProjectGroupEntry> {
        self.entries
            .iter()
            .filter_map(|entry| match entry {
                SidebarEntry::ProjectHeader(group) => Some(group.clone()),
                _ => None,
            })
            .collect()
    }

    fn refresh_after_structure_change(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) {
        let project_groups = self.current_project_groups();
        self.rebuild_entries(project_groups, cx);
        let query = self.query.clone();
        cx.spawn_in(window, async move |picker, cx| {
            picker
                .update_in(cx, |picker, window, cx| {
                    picker.update_matches(query, window, cx);
                })
                .log_err();
        })
        .detach();
    }

    fn toggle_group_collapsed(
        &mut self,
        group_key: &ProjectGroupKey,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) {
        if self.collapsed_groups.contains(group_key) {
            self.collapsed_groups.remove(group_key);
        } else {
            self.collapsed_groups.insert(group_key.clone());
        }
        self.refresh_after_structure_change(window, cx);
    }

    fn serialized_state(&self) -> SerializedSidebarState {
        SerializedSidebarState {
            width: None,
            active_view: SerializedSidebarView::ThreadList,
            collapsed_groups: self
                .collapsed_groups
                .iter()
                .map(|group_key| {
                    group_key
                        .path_list()
                        .paths()
                        .iter()
                        .map(|path| path.to_path_buf())
                        .collect()
                })
                .collect(),
            expanded_groups: self
                .expanded_groups
                .iter()
                .map(|(group_key, batches)| {
                    (
                        group_key
                            .path_list()
                            .paths()
                            .iter()
                            .map(|path| path.to_path_buf())
                            .collect(),
                        *batches,
                    )
                })
                .collect(),
        }
    }

    fn group_for_entry(entry: &SidebarEntry) -> Option<ProjectGroupEntry> {
        match entry {
            SidebarEntry::ProjectHeader(group)
            | SidebarEntry::ProjectNewThread(group) => Some(group.clone()),
            SidebarEntry::ProjectDraftThread { group, .. }
            | SidebarEntry::ProjectThread { group, .. }
            | SidebarEntry::ProjectViewMore { group, .. } => Some(group.clone()),
            SidebarEntry::Separator(_) | SidebarEntry::RecentProject(_) => None,
        }
    }

    fn selected_project_group(&self) -> Option<ProjectGroupEntry> {
        self.matches
            .get(self.selected_index)
            .and_then(|matched| Self::group_for_entry(&matched.entry))
            .or_else(|| {
                self.active_group_key.as_ref().and_then(|active_key| {
                    self.entries.iter().find_map(|entry| {
                        let group = Self::group_for_entry(entry)?;
                        (&group.key == active_key).then_some(group)
                    })
                })
            })
    }

    fn activate_group_workspace(
        &self,
        group: &ProjectGroupEntry,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Option<Entity<Workspace>> {
        let target_index = group.activation_index;
        let target_workspace = self
            .multi_workspace
            .read(cx)
            .workspaces()
            .get(target_index)
            .cloned();
        self.multi_workspace.update(cx, |multi_workspace, cx| {
            multi_workspace.activate_index(target_index, window, cx);
        });
        target_workspace
    }

    fn open_new_thread_in_group(
        &mut self,
        group: &ProjectGroupEntry,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) {
        if let Some(workspace) = self.activate_group_workspace(group, window, cx) {
            workspace.update(cx, |workspace, cx| {
                workspace.focus_panel::<AgentPanel>(window, cx);
            });
            window.dispatch_action(NewThread.boxed_clone(), cx);
        }
    }

    fn set_recent_projects(&mut self, recent_projects: Vec<RecentProjectEntry>, cx: &App) {
        self.recent_project_thread_titles.clear();
        if let Some(map) = read_thread_title_map() {
            for entry in &recent_projects {
                let path_key = sorted_paths_key(&entry.paths);
                if let Some(title) = map.get(&path_key) {
                    self.recent_project_thread_titles
                        .insert(entry.full_path.clone(), title.clone().into());
                }
            }
        }

        self.recent_projects = recent_projects;

        let project_groups: Vec<ProjectGroupEntry> = self
            .entries
            .iter()
            .filter_map(|entry| match entry {
                SidebarEntry::ProjectHeader(thread) => Some(thread.clone()),
                _ => None,
            })
            .collect();
        self.rebuild_entries(project_groups, cx);
    }

    fn open_project_group_path_sets(&self, cx: &App) -> Vec<Vec<PathBuf>> {
        self.multi_workspace
            .read(cx)
            .project_groups(cx)
            .map(|(key, _)| {
                let mut paths = key
                    .path_list()
                    .paths()
                    .iter()
                    .map(|path| path.clone())
                    .collect::<Vec<_>>();
                paths.sort();
                paths
            })
            .collect()
    }

    fn rebuild_entries(&mut self, project_groups: Vec<ProjectGroupEntry>, cx: &App) {
        let open_path_sets = self.open_project_group_path_sets(cx);

        self.entries.clear();

        if !project_groups.is_empty() {
            self.entries
                .push(SidebarEntry::Separator("Projects".into()));
            for group in project_groups {
                self.entries.push(SidebarEntry::ProjectHeader(group.clone()));
                if self.collapsed_groups.contains(&group.key) {
                    continue;
                }

                let total_threads = group.threads.len();
                if let Some(draft_title) = group.draft_text.clone() {
                    self.entries.push(SidebarEntry::ProjectDraftThread {
                        group: group.clone(),
                        title: draft_title,
                    });
                }
                if total_threads == 0 {
                    if group.draft_text.is_none() {
                        self.entries.push(SidebarEntry::ProjectNewThread(group.clone()));
                    }
                    continue;
                }

                let visible_threads = if self.query.is_empty() {
                    let extra_batches = self.expanded_groups.get(&group.key).copied().unwrap_or(0);
                    (DEFAULT_THREADS_SHOWN * (extra_batches + 1)).min(total_threads)
                } else {
                    total_threads
                };

                let mut visible = Vec::new();
                let mut promoted = Vec::new();
                for (index, thread) in group.threads.iter().cloned().enumerate() {
                    let within_limit = index < visible_threads;
                    let should_promote =
                        thread.status == AgentThreadStatus::Running || thread.is_active;
                    if within_limit {
                        visible.push(thread);
                    } else if should_promote {
                        promoted.push(thread);
                    }
                }

                let shown = visible.len() + promoted.len();
                for thread in visible.into_iter().chain(promoted.into_iter()) {
                    self.entries.push(SidebarEntry::ProjectThread {
                        group: group.clone(),
                        thread,
                    });
                }

                if self.query.is_empty() && total_threads > DEFAULT_THREADS_SHOWN {
                    self.entries.push(SidebarEntry::ProjectViewMore {
                        group: group.clone(),
                        shown,
                        total: total_threads,
                        is_fully_expanded: shown >= total_threads,
                    });
                }
            }
        }

        let recent: Vec<_> = self
            .recent_projects
            .iter()
            .filter(|project| {
                let mut project_paths: Vec<&Path> =
                    project.paths.iter().map(|p| p.as_path()).collect();
                project_paths.sort();
                !open_path_sets.iter().any(|open_paths| {
                    open_paths.len() == project_paths.len()
                        && open_paths
                            .iter()
                            .zip(&project_paths)
                            .all(|(a, b)| a.as_path() == *b)
                })
            })
            .cloned()
            .collect();

        if !recent.is_empty() {
            self.entries
                .push(SidebarEntry::Separator("Recent Projects".into()));
            for project in recent {
                self.entries.push(SidebarEntry::RecentProject(project));
            }
        }
    }

    fn open_recent_project(paths: Vec<PathBuf>, window: &mut Window, cx: &mut App) {
        let Some(handle) = window.window_handle().downcast::<MultiWorkspace>() else {
            return;
        };

        cx.defer(move |cx| {
            if let Some(task) = handle
                .update(cx, |multi_workspace, window, cx| {
                    multi_workspace.open_project(paths, OpenMode::Activate, window, cx)
                })
                .log_err()
            {
                task.detach_and_log_err(cx);
            }
        });
    }
}

impl PickerDelegate for WorkspacePickerDelegate {
    type ListItem = AnyElement;

    fn match_count(&self) -> usize {
        self.matches.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(
        &mut self,
        ix: usize,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) {
        self.selected_index = ix;
    }

    fn can_select(
        &mut self,
        ix: usize,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> bool {
        match self.matches.get(ix) {
            Some(SidebarMatch {
                entry: SidebarEntry::Separator(_) | SidebarEntry::ProjectHeader(_),
                ..
            }) => false,
            _ => true,
        }
    }

    fn placeholder_text(&self, _window: &mut Window, _cx: &mut App) -> Arc<str> {
        "Search…".into()
    }

    fn no_matches_text(&self, _window: &mut Window, _cx: &mut App) -> Option<SharedString> {
        if self.query.is_empty() {
            None
        } else {
            Some("No threads match your search.".into())
        }
    }

    fn update_matches(
        &mut self,
        query: String,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        let query_changed = self.query != query;
        self.query = query.clone();
        if query_changed {
            self.hovered_thread_item = None;
        }
        let entries = self.entries.clone();

        if query.is_empty() {
            let active_index = self
                .active_group_key
                .as_ref()
                .and_then(|active_key| {
                    entries.iter().position(|entry| {
                        matches!(entry, SidebarEntry::ProjectThread { group, .. } if &group.key == active_key)
                    })
                })
                .or_else(|| {
                    self.active_group_key.as_ref().and_then(|active_key| {
                        entries.iter().position(|entry| {
                            matches!(entry, SidebarEntry::ProjectHeader(group) if &group.key == active_key)
                        })
                    })
                })
                .unwrap_or(0);

            self.matches = entries
                .into_iter()
                .map(|entry| SidebarMatch {
                    entry,
                    positions: Vec::new(),
                })
                .collect();

            self.selected_index = active_index.min(self.matches.len().saturating_sub(1));
            return Task::ready(());
        }

        let executor = cx.background_executor().clone();
        cx.spawn_in(window, async move |picker, cx| {
            let matches = cx
                .background_spawn(async move {
                    let data_entries: Vec<(usize, &SidebarEntry)> = entries
                        .iter()
                        .enumerate()
                        .filter(|(_, entry)| {
                            !matches!(
                                entry,
                                SidebarEntry::Separator(_) | SidebarEntry::ProjectViewMore { .. }
                            )
                        })
                        .collect();

                    let candidates: Vec<StringMatchCandidate> = data_entries
                        .iter()
                        .enumerate()
                        .map(|(candidate_index, (_, entry))| {
                            StringMatchCandidate::new(candidate_index, entry.searchable_text())
                        })
                        .collect();

                    let search_matches = fuzzy::match_strings(
                        &candidates,
                        &query,
                        false,
                        true,
                        MAX_MATCHES,
                        &Default::default(),
                        executor,
                    )
                    .await;

                    let mut workspace_matches = Vec::new();
                    let mut project_matches = Vec::new();

                    for search_match in search_matches {
                        let (original_index, _) = data_entries[search_match.candidate_id];
                        let entry = entries[original_index].clone();
                    let sidebar_match = SidebarMatch {
                        positions: search_match.positions,
                        entry: entry.clone(),
                    };
                    match entry {
                            SidebarEntry::ProjectHeader(_)
                            | SidebarEntry::ProjectDraftThread { .. }
                            | SidebarEntry::ProjectThread { .. } => {
                                workspace_matches.push(sidebar_match)
                            }
                            SidebarEntry::ProjectViewMore { .. } => {}
                            SidebarEntry::ProjectNewThread(_) => workspace_matches.push(sidebar_match),
                            SidebarEntry::RecentProject(_) => project_matches.push(sidebar_match),
                            SidebarEntry::Separator(_) => {}
                        }
                    }

                    let mut result = Vec::new();
                    if !workspace_matches.is_empty() {
                        result.push(SidebarMatch {
                            entry: SidebarEntry::Separator("Projects".into()),
                            positions: Vec::new(),
                        });
                        result.extend(workspace_matches);
                    }
                    if !project_matches.is_empty() {
                        result.push(SidebarMatch {
                            entry: SidebarEntry::Separator("Recent Projects".into()),
                            positions: Vec::new(),
                        });
                        result.extend(project_matches);
                    }
                    result
                })
                .await;

            picker
                .update_in(cx, |picker, _window, _cx| {
                    picker.delegate.matches = matches;
                    if picker.delegate.matches.is_empty() {
                        picker.delegate.selected_index = 0;
                    } else {
                        let first_selectable = picker
                            .delegate
                            .matches
                            .iter()
                            .position(|m| !matches!(m.entry, SidebarEntry::Separator(_)))
                            .unwrap_or(0);
                        picker.delegate.selected_index = first_selectable;
                    }
                })
                .log_err();
        })
    }

    fn confirm(&mut self, _secondary: bool, window: &mut Window, cx: &mut Context<Picker<Self>>) {
        let Some(selected_entry) = self
            .matches
            .get(self.selected_index)
            .map(|sidebar_match| sidebar_match.entry.clone())
        else {
            return;
        };

        match selected_entry {
            SidebarEntry::Separator(_) => {}
            SidebarEntry::ProjectHeader(thread_entry) => {
                let target_index = thread_entry.activation_index;
                self.multi_workspace.update(cx, |multi_workspace, cx| {
                    multi_workspace.activate_index(target_index, window, cx);
                });
            }
            SidebarEntry::ProjectDraftThread { group, .. } => {
                if let Some(workspace) = self.activate_group_workspace(&group, window, cx) {
                    workspace.update(cx, |workspace, cx| {
                        workspace.focus_panel::<AgentPanel>(window, cx);
                    });
                }
            }
            SidebarEntry::ProjectThread { group, thread } => {
                if let Some(workspace) = self.activate_group_workspace(&group, window, cx) {
                    let mut agent_thread = AgentSessionInfo::new(thread.metadata.session_id.clone());
                    agent_thread.cwd = thread.metadata.folder_paths.paths().first().cloned();
                    agent_thread.title = Some(thread.metadata.title.clone());
                    agent_thread.updated_at = Some(thread.metadata.updated_at);

                    let agent = match thread.metadata.agent_id.as_ref() {
                        "Zed Agent" => agent_ui::ExternalAgent::NativeAgent,
                        GEMINI_NAME => agent_ui::ExternalAgent::Gemini,
                        CLAUDE_CODE_NAME => agent_ui::ExternalAgent::ClaudeCode,
                        CODEX_NAME => agent_ui::ExternalAgent::Codex,
                        name => agent_ui::ExternalAgent::Custom {
                            name: name.to_string().into(),
                        },
                    };

                    if let Some(panel) = workspace.read(cx).panel::<AgentPanel>(cx) {
                        panel.update(cx, |panel, cx| {
                            panel.open_thread_with_agent(agent, agent_thread, window, cx);
                        });
                    }
                    workspace.update(cx, |workspace, cx| {
                        workspace.focus_panel::<AgentPanel>(window, cx);
                    });
                }
            }
            SidebarEntry::ProjectViewMore {
                group,
                is_fully_expanded,
                ..
            } => {
                if is_fully_expanded {
                    self.expanded_groups.remove(&group.key);
                } else {
                    let current = self.expanded_groups.get(&group.key).copied().unwrap_or(0);
                    self.expanded_groups.insert(group.key.clone(), current + 1);
                }
                self.refresh_after_structure_change(window, cx);
            }
            SidebarEntry::ProjectNewThread(group) => {
                self.open_new_thread_in_group(&group, window, cx);
            }
            SidebarEntry::RecentProject(project_entry) => {
                let paths = project_entry.paths.clone();
                Self::open_recent_project(paths, window, cx);
            }
        }
    }

    fn dismissed(&mut self, _window: &mut Window, _cx: &mut Context<Picker<Self>>) {}

    fn render_match(
        &self,
        index: usize,
        selected: bool,
        _window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        let match_entry = self.matches.get(index)?;
        let SidebarMatch { entry, positions } = match_entry;

        match entry {
            SidebarEntry::Separator(title) => Some(
                v_flex()
                    .when(index > 0, |this| {
                        this.mt_1()
                            .gap_2()
                            .child(Divider::horizontal().color(DividerColor::BorderFaded))
                    })
                    .child(ListSubHeader::new(title.clone()).inset(true))
                    .into_any_element(),
            ),
            SidebarEntry::ProjectHeader(group_entry) => Some(
                {
                    let picker = cx.entity().downgrade();
                    let disclosure_picker = picker.clone();
                    let group_key = group_entry.key.clone();
                    let new_thread_group = group_entry.clone();
                    let is_active_group = self
                        .active_group_key
                        .as_ref()
                        .is_some_and(|active_key| active_key == &group_entry.key);
                    let workspace_count = self.multi_workspace.read(cx).workspaces().len();
                    v_flex()
                        .when(index > 0, |this| this.mt_1())
                        .child(
                            ListSubHeader::new(group_entry.worktree_label.clone())
                                .left_icon(Some(if group_entry.is_remote {
                                    IconName::Server
                                } else {
                                    IconName::Folder
                                }))
                                .inset(true)
                                .toggle_state(is_active_group)
                                .end_slot(
                                    h_flex()
                                        .gap_1()
                                        .child(
                                            Label::new(group_entry.thread_count().to_string())
                                                .size(LabelSize::XSmall)
                                                .color(Color::Muted),
                                        )
                                        .when(group_entry.has_running_threads, |this| {
                                            this.child(
                                                Icon::new(IconName::LoadCircle)
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted)
                                                    .with_rotate_animation(2),
                                            )
                                        })
                                        .when(
                                            workspace_count > 1
                                                && group_entry.workspace_indices.len() == 1,
                                            |this| {
                                                let multi_workspace = self.multi_workspace.clone();
                                                let activation_index = group_entry.activation_index;
                                                this.child(
                                                    IconButton::new(
                                                        SharedString::from(format!(
                                                            "remove-group-workspace-{}",
                                                            activation_index
                                                        )),
                                                        IconName::Close,
                                                    )
                                                    .icon_size(IconSize::XSmall)
                                                    .icon_color(Color::Muted)
                                                    .tooltip(Tooltip::text("Remove Workspace"))
                                                    .on_click(move |_, window, cx| {
                                                        multi_workspace.update(cx, |mw, cx| {
                                                            mw.remove_workspace(
                                                                activation_index,
                                                                window,
                                                                cx,
                                                            );
                                                        });
                                                    }),
                                                )
                                            },
                                        )
                                        .child(
                                            IconButton::new(
                                                SharedString::from(format!(
                                                    "new-thread-in-group-{}",
                                                    group_entry.activation_index
                                                )),
                                                IconName::Thread,
                                            )
                                            .icon_size(IconSize::XSmall)
                                            .tooltip(|_window, cx| {
                                                Tooltip::for_action(
                                                    "New Thread In Project",
                                                    &NewThreadInGroup,
                                                    cx,
                                                )
                                            })
                                            .on_click(move |_, window, cx| {
                                                if let Some(picker) = picker.upgrade() {
                                                    let group = new_thread_group.clone();
                                                    picker.update(cx, |picker, cx| {
                                                        picker.delegate.open_new_thread_in_group(
                                                            &group, window, cx,
                                                        );
                                                    });
                                                }
                                            }),
                                        )
                                        .child(
                                            Disclosure::new(
                                                SharedString::from(format!(
                                                    "collapse-group-{}",
                                                    group_entry.activation_index
                                                )),
                                                !self.collapsed_groups.contains(&group_entry.key),
                                            )
                                            .on_click(move |_, window, cx| {
                                                if let Some(picker) = disclosure_picker.upgrade() {
                                                    picker.update(cx, |picker, cx| {
                                                        picker.delegate.toggle_group_collapsed(
                                                            &group_key, window, cx,
                                                        );
                                                    });
                                                }
                                            }),
                                        )
                                        .into_any_element(),
                                ),
                        )
                        .into_any_element()
                },
            ),
            SidebarEntry::ProjectDraftThread { group, title } => Some(
                ThreadItem::new(
                    SharedString::from(format!("workspace-draft-thread-{}", group.activation_index)),
                    title.clone(),
                )
                .icon(IconName::ZedAgent)
                .generating_title(title.as_ref() == "New Thread…")
                .selected(selected)
                .timestamp("Draft")
                .worktree(group.worktree_label.clone())
                .worktree_highlight_positions(positions.clone())
                .into_any_element(),
            ),
            SidebarEntry::ProjectThread { group, thread } => {
                let worktree_label = thread.worktree_label.clone();
                let full_path = thread.full_path.clone();
                let activation_index = group.activation_index;
                let group_key = group.key.clone();
                let multi_workspace = self.multi_workspace.clone();
                let workspace_count = self.multi_workspace.read(cx).workspaces().len();
                let is_hovered = self
                    .hovered_thread_item
                    .as_ref()
                    .is_some_and(|hovered| hovered == &group_key);

                let remove_btn = IconButton::new(
                    format!("remove-workspace-{}", activation_index),
                    IconName::Close,
                )
                .icon_size(IconSize::Small)
                .icon_color(Color::Muted)
                .tooltip(Tooltip::text("Remove Workspace"))
                .on_click({
                    let multi_workspace = multi_workspace;
                    move |_, window, cx| {
                        multi_workspace.update(cx, |mw, cx| {
                            mw.remove_workspace(activation_index, window, cx);
                        });
                    }
                });

                let has_notification = self.notified_groups.contains(&group_key);
                let generating_title = thread.generating_title;
                let running = thread.status == AgentThreadStatus::Running;

                Some(
                    ThreadItem::new(
                        SharedString::from(format!(
                            "workspace-item-{}-{}",
                            activation_index, thread.metadata.session_id.0
                        )),
                        thread.metadata.title.clone(),
                    )
                    .icon(thread.icon)
                    .running(running)
                    .generation_done(has_notification)
                    .generating_title(generating_title)
                    .selected(selected)
                    .timestamp(format_thread_timestamp(thread.metadata.updated_at))
                    .worktree(worktree_label.clone())
                    .worktree_highlight_positions(positions.clone())
                    .when(workspace_count > 1 && group.workspace_indices.len() == 1, |item| {
                        item.action_slot(remove_btn)
                    })
                    .hovered(is_hovered)
                    .on_hover(cx.listener(move |picker, is_hovered, _window, cx| {
                        let mut changed = false;
                        if *is_hovered {
                            if picker.delegate.hovered_thread_item.as_ref() != Some(&group_key) {
                                picker.delegate.hovered_thread_item = Some(group_key.clone());
                                changed = true;
                            }
                        } else if picker.delegate.hovered_thread_item.as_ref() == Some(&group_key) {
                            picker.delegate.hovered_thread_item = None;
                            changed = true;
                        }
                        if changed {
                            cx.notify();
                        }
                    }))
                    .when(!full_path.is_empty(), |this| {
                        this.tooltip(move |_, cx| {
                            Tooltip::with_meta(worktree_label.clone(), None, full_path.clone(), cx)
                        })
                    })
                    .into_any_element(),
                )
            }
            SidebarEntry::ProjectViewMore {
                group,
                shown,
                total,
                is_fully_expanded,
            } => {
                let label: SharedString = if *is_fully_expanded {
                    "Show Less".into()
                } else {
                    format!("View More ({})", total.saturating_sub(*shown)).into()
                };

                Some(
                    ThreadItem::new(
                        SharedString::from(format!(
                            "workspace-view-more-{}",
                            group.activation_index
                        )),
                        label,
                    )
                    .icon(if *is_fully_expanded {
                        IconName::ChevronUp
                    } else {
                        IconName::ChevronDown
                    })
                    .selected(selected)
                    .worktree(group.worktree_label.clone())
                    .into_any_element(),
                )
            }
            SidebarEntry::ProjectNewThread(group) => Some(
                ThreadItem::new(
                    SharedString::from(format!("workspace-new-thread-{}", group.activation_index)),
                    "New Thread",
                )
                .icon(IconName::Plus)
                .selected(selected)
                .timestamp("Create")
                .worktree(group.worktree_label.clone())
                .worktree_highlight_positions(positions.clone())
                .into_any_element(),
            ),
            SidebarEntry::RecentProject(project_entry) => {
                let name = project_entry.name.clone();
                let full_path = project_entry.full_path.clone();
                let item_id: SharedString =
                    format!("recent-project-{:?}", project_entry.workspace_id).into();

                Some(
                    ThreadItem::new(item_id, name.clone())
                        .icon(IconName::Folder)
                        .selected(selected)
                        .highlight_positions(positions.clone())
                        .tooltip(move |_, cx| {
                            Tooltip::with_meta(name.clone(), None, full_path.clone(), cx)
                        })
                        .into_any_element(),
                )
            }
        }
    }

    fn render_editor(
        &self,
        editor: &Arc<dyn ErasedEditor>,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Div {
        h_flex()
            .h(Tab::container_height(cx))
            .w_full()
            .px_2()
            .gap_2()
            .justify_between()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(
                Icon::new(IconName::MagnifyingGlass)
                    .color(Color::Muted)
                    .size(IconSize::Small),
            )
            .child(editor.render(window, cx))
    }
}

pub struct Sidebar {
    multi_workspace: Entity<MultiWorkspace>,
    width: Pixels,
    picker: Entity<Picker<WorkspacePickerDelegate>>,
    _subscription: Subscription,
    _project_subscriptions: Vec<Subscription>,
    _agent_panel_subscriptions: Vec<Subscription>,
    _thread_subscriptions: Vec<Subscription>,
    _thread_switcher_subscriptions: Vec<Subscription>,
    view: SidebarView,
    _archive_subscription: Option<Subscription>,
    thread_switcher: Option<Entity<ThreadSwitcher>>,
    #[cfg(any(test, feature = "test-support"))]
    test_thread_infos: HashMap<usize, AgentThreadInfo>,
    #[cfg(any(test, feature = "test-support"))]
    test_recent_project_thread_titles: HashMap<SharedString, SharedString>,
    _fetch_recent_projects: Task<()>,
}

impl EventEmitter<SidebarEvent> for Sidebar {}

impl Sidebar {
    pub fn new(
        multi_workspace: Entity<MultiWorkspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let delegate = WorkspacePickerDelegate::new(multi_workspace.clone());
        let picker = cx.new(|cx| {
            Picker::list(delegate, window, cx)
                .max_height(None)
                .show_scrollbar(true)
                .modal(false)
        });

        let subscription = cx.observe_in(
            &multi_workspace,
            window,
            |this, multi_workspace, window, cx| {
                this.queue_refresh(multi_workspace, window, cx);
            },
        );

        let fetch_recent_projects = {
            let picker = picker.downgrade();
            let fs = <dyn Fs>::global(cx);
            cx.spawn_in(window, async move |_this, cx| {
                let projects = get_recent_projects(None, None, fs).await;

                cx.update(|window, cx| {
                    if let Some(picker) = picker.upgrade() {
                        picker.update(cx, |picker, cx| {
                            picker.delegate.set_recent_projects(projects, cx);
                            let query = picker.query(cx);
                            picker.update_matches(query, window, cx);
                        });
                    }
                })
                .log_err();
            })
        };

        let mut this = Self {
            multi_workspace,
            width: DEFAULT_WIDTH,
            picker,
            _subscription: subscription,
            _project_subscriptions: Vec::new(),
            _agent_panel_subscriptions: Vec::new(),
            _thread_subscriptions: Vec::new(),
            _thread_switcher_subscriptions: Vec::new(),
            view: SidebarView::ThreadList,
            _archive_subscription: None,
            thread_switcher: None,
            #[cfg(any(test, feature = "test-support"))]
            test_thread_infos: HashMap::new(),
            #[cfg(any(test, feature = "test-support"))]
            test_recent_project_thread_titles: HashMap::new(),
            _fetch_recent_projects: fetch_recent_projects,
        };
        this.queue_refresh(this.multi_workspace.clone(), window, cx);
        this
    }

    fn subscribe_to_projects(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<Subscription> {
        let projects: Vec<_> = self
            .multi_workspace
            .read(cx)
            .workspaces()
            .iter()
            .map(|w| w.read(cx).project().clone())
            .collect();

        projects
            .iter()
            .map(|project| {
                cx.subscribe_in(
                    project,
                    window,
                    |this, _project, event, window, cx| match event {
                        ProjectEvent::WorktreeAdded(_)
                        | ProjectEvent::WorktreeRemoved(_)
                        | ProjectEvent::WorktreeOrderChanged => {
                            this.queue_refresh(this.multi_workspace.clone(), window, cx);
                        }
                        _ => {}
                    },
                )
            })
            .collect()
    }

    fn build_project_group_entries(
        &self,
        multi_workspace: &MultiWorkspace,
        cx: &App,
    ) -> (Vec<ProjectGroupEntry>, Option<ProjectGroupKey>) {
        let persisted_titles = read_thread_title_map().unwrap_or_default();
        let active_workspace_index = multi_workspace.active_workspace_index();

        #[allow(unused_mut)]
        let mut entries: Vec<ProjectGroupEntry> = multi_workspace
            .project_groups(cx)
            .map(|(key, workspaces)| {
                let indexed_workspaces = workspaces
                    .into_iter()
                    .filter_map(|workspace| {
                        multi_workspace
                            .workspaces()
                            .iter()
                            .position(|candidate| candidate == &workspace)
                            .map(|index| (index, workspace))
                    })
                    .collect::<Vec<_>>();
                ProjectGroupEntry::new(
                    key,
                    &indexed_workspaces,
                    active_workspace_index,
                    &persisted_titles,
                    cx,
                )
            })
            .collect();

        #[cfg(any(test, feature = "test-support"))]
        for (index, info) in &self.test_thread_infos {
            if let Some(entry) = entries.get_mut(*index) {
                entry.thread_info = Some(info.clone());
            }
        }

        let active_group_key = multi_workspace
            .workspaces()
            .get(active_workspace_index)
            .map(|workspace| workspace.read(cx).project_group_key(cx));

        (entries, active_group_key)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn set_test_recent_projects(
        &self,
        projects: Vec<RecentProjectEntry>,
        cx: &mut Context<Self>,
    ) {
        self.picker.update(cx, |picker, _cx| {
            picker.delegate.recent_projects = projects;
        });
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn set_test_thread_info(
        &mut self,
        index: usize,
        title: SharedString,
        status: AgentThreadStatus,
    ) {
        self.test_thread_infos.insert(
            index,
            AgentThreadInfo {
                title,
                status,
                generating_title: false,
            },
        );
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn set_test_recent_project_thread_title(
        &mut self,
        full_path: SharedString,
        title: SharedString,
        cx: &mut Context<Self>,
    ) {
        self.test_recent_project_thread_titles
            .insert(full_path.clone(), title.clone());
        self.picker.update(cx, |picker, _cx| {
            picker
                .delegate
                .recent_project_thread_titles
                .insert(full_path, title);
        });
    }

    fn subscribe_to_agent_panels(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<Subscription> {
        let workspaces: Vec<_> = self.multi_workspace.read(cx).workspaces().to_vec();

        workspaces
            .iter()
            .map(|workspace| {
                if let Some(agent_panel) = workspace.read(cx).panel::<AgentPanel>(cx) {
                    cx.subscribe_in(
                        &agent_panel,
                        window,
                        |this, _, _event: &AgentPanelEvent, window, cx| {
                            this.queue_refresh(this.multi_workspace.clone(), window, cx);
                        },
                    )
                } else {
                    // Panel hasn't loaded yet — observe the workspace so we
                    // re-subscribe once the panel appears on its dock.
                    cx.observe_in(workspace, window, |this, _, window, cx| {
                        this.queue_refresh(this.multi_workspace.clone(), window, cx);
                    })
                }
            })
            .collect()
    }

    fn subscribe_to_threads(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<Subscription> {
        let workspaces: Vec<_> = self.multi_workspace.read(cx).workspaces().to_vec();

        workspaces
            .iter()
            .filter_map(|workspace| {
                let agent_panel = workspace.read(cx).panel::<AgentPanel>(cx)?;
                let thread = agent_panel.read(cx).active_agent_thread(cx)?;
                Some(cx.observe_in(&thread, window, |this, _, window, cx| {
                    this.queue_refresh(this.multi_workspace.clone(), window, cx);
                }))
            })
            .collect()
    }

    fn persist_thread_titles(
        &self,
        entries: &[ProjectGroupEntry],
        _multi_workspace: &Entity<MultiWorkspace>,
        cx: &mut Context<Self>,
    ) {
        let mut map = read_thread_title_map().unwrap_or_default();
        let mut changed = false;

        for entry in entries {
            if let Some(ref info) = entry.thread_info {
                let paths = entry.key.path_list().paths();
                if paths.is_empty() {
                    continue;
                }
                let path_key = sorted_paths_key(paths);
                let title = info.title.to_string();
                if map.get(&path_key) != Some(&title) {
                    map.insert(path_key, title);
                    changed = true;
                }
            }
        }

        if changed {
            if let Some(json) = serde_json::to_string(&map).log_err() {
                cx.background_spawn(async move {
                    KEY_VALUE_STORE
                        .write_kvp(LAST_THREAD_TITLES_KEY.into(), json)
                        .await
                        .log_err();
                })
                .detach();
            }
        }
    }

    fn queue_refresh(
        &mut self,
        multi_workspace: Entity<MultiWorkspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.defer_in(window, move |this, window, cx| {
            this._project_subscriptions = this.subscribe_to_projects(window, cx);
            this._agent_panel_subscriptions = this.subscribe_to_agent_panels(window, cx);
            this._thread_subscriptions = this.subscribe_to_threads(window, cx);
            let (entries, active_group_key) = multi_workspace.read_with(cx, |multi_workspace, cx| {
                this.build_project_group_entries(multi_workspace, cx)
            });

            this.persist_thread_titles(&entries, &multi_workspace, cx);

            let had_notifications = !this.picker.read(cx).delegate.notified_groups.is_empty();
            this.picker.update(cx, |picker, cx| {
                picker.delegate.set_entries(entries, active_group_key, cx);
                let query = picker.query(cx);
                picker.update_matches(query, window, cx);
            });
            let has_notifications = !this.picker.read(cx).delegate.notified_groups.is_empty();
            if had_notifications != has_notifications {
                multi_workspace.update(cx, |_, cx| cx.notify());
            }
        });
    }

    fn toggle_archive(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.view {
            SidebarView::ThreadList => self.show_archive(window, cx),
            SidebarView::Archive(_) => self.show_thread_list(cx),
        }
    }

    fn format_switcher_timestamp() -> SharedString {
        "".into()
    }

    fn mru_threads_for_switcher(&self, cx: &App) -> Vec<ThreadSwitcherEntry> {
        let groups = self.picker.read(cx).delegate.current_project_groups();
        let workspaces = self.multi_workspace.read(cx).workspaces().to_vec();
        let notified = self.picker.read(cx).delegate.notified_groups.clone();

        let mut entries = groups
            .into_iter()
            .flat_map(|group| {
                let workspace = workspaces.get(group.activation_index).cloned();
                let project_name = group.worktree_label.clone();
                let is_notified = notified.contains(&group.key);
                group.threads.into_iter().filter_map(move |thread| {
                    let workspace = workspace.clone()?;
                    Some(ThreadSwitcherEntry {
                        session_id: thread.metadata.session_id.clone(),
                        title: thread.metadata.title.clone(),
                        icon: thread.icon,
                        status: thread.status,
                        workspace,
                        project_name: project_name.clone(),
                        worktree_label: thread.worktree_label.clone(),
                        generating_title: thread.generating_title,
                        notified: is_notified,
                        timestamp: Self::format_switcher_timestamp(),
                    })
                })
            })
            .collect::<Vec<_>>();

        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries
    }

    fn open_thread_metadata_in_workspace(
        workspace: &Entity<Workspace>,
        session_id: &acp::SessionId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let store = ThreadMetadataStore::global(cx);
        let Some(metadata) = ({
            let store = store.read(cx);
            store.entry(session_id).cloned()
        }) else {
            return;
        };

        let mut thread = AgentSessionInfo::new(metadata.session_id.clone());
        thread.cwd = Self::archived_thread_paths(&metadata).first().cloned();
        thread.title = Some(metadata.title.clone());
        thread.updated_at = Some(metadata.updated_at);

        let agent = match metadata.agent_id.as_ref() {
            "Zed Agent" => agent_ui::ExternalAgent::NativeAgent,
            GEMINI_NAME => agent_ui::ExternalAgent::Gemini,
            CLAUDE_CODE_NAME => agent_ui::ExternalAgent::ClaudeCode,
            CODEX_NAME => agent_ui::ExternalAgent::Codex,
            name => agent_ui::ExternalAgent::Custom {
                name: name.to_string().into(),
            },
        };

        if let Some(panel) = workspace.read(cx).panel::<AgentPanel>(cx) {
            panel.update(cx, |panel, cx| {
                panel.open_thread_with_agent(agent, thread, window, cx);
            });
        }
    }

    fn dismiss_thread_switcher(&mut self) {
        self.thread_switcher = None;
        self._thread_switcher_subscriptions.clear();
    }

    fn on_toggle_thread_switcher(
        &mut self,
        _: &ToggleThreadSwitcher,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(thread_switcher) = &self.thread_switcher {
            thread_switcher.update(cx, |switcher, cx| {
                switcher.cycle_selection(cx);
            });
            return;
        }

        let entries = self.mru_threads_for_switcher(cx);
        if entries.len() < 2 {
            return;
        }

        let original_workspace = self
            .multi_workspace
            .read(cx)
            .workspaces()
            .get(self.multi_workspace.read(cx).active_workspace_index())
            .cloned();
        let original_session_id = original_workspace.as_ref().and_then(|workspace| {
            let agent_panel = workspace.read(cx).panel::<AgentPanel>(cx)?;
            let thread = agent_panel.read(cx).active_agent_thread(cx)?;
            Some(thread.read(cx).session_id().clone())
        });

        let Some(active_workspace) = original_workspace.clone() else {
            return;
        };

        active_workspace.update(cx, |workspace, cx| {
            workspace.toggle_modal(window, cx, |window, cx| {
                ThreadSwitcher::new(entries, false, window, cx)
            });
        });

        let Some(thread_switcher) = active_workspace.read(cx).active_modal::<ThreadSwitcher>(cx) else {
            return;
        };

        let mut subscriptions = Vec::new();
        subscriptions.push(cx.subscribe_in(
            &thread_switcher,
            window,
            move |this, _, event: &ThreadSwitcherEvent, window, cx| match event {
                ThreadSwitcherEvent::Preview { session_id, workspace } => {
                    this.multi_workspace.update(cx, |mw, cx| {
                        mw.activate(workspace.clone(), cx);
                    });
                    Self::open_thread_metadata_in_workspace(workspace, session_id, window, cx);
                }
                ThreadSwitcherEvent::Confirmed { session_id, workspace } => {
                    this.multi_workspace.update(cx, |mw, cx| {
                        mw.activate(workspace.clone(), cx);
                    });
                    Self::open_thread_metadata_in_workspace(workspace, session_id, window, cx);
                    workspace.update(cx, |workspace, cx| {
                        workspace.focus_panel::<AgentPanel>(window, cx);
                    });
                    this.dismiss_thread_switcher();
                }
                ThreadSwitcherEvent::Dismissed => {
                    if let (Some(workspace), Some(session_id)) =
                        (original_workspace.clone(), original_session_id.clone())
                    {
                        this.multi_workspace.update(cx, |mw, cx| {
                            mw.activate(workspace.clone(), cx);
                        });
                        Self::open_thread_metadata_in_workspace(&workspace, &session_id, window, cx);
                    }
                    this.dismiss_thread_switcher();
                }
            },
        ));
        subscriptions.push(cx.subscribe_in(
            &thread_switcher,
            window,
            |this, _, _: &gpui::DismissEvent, _window, _cx| {
                this.dismiss_thread_switcher();
            },
        ));

        self.thread_switcher = Some(thread_switcher);
        self._thread_switcher_subscriptions = subscriptions;
    }

    fn selected_project_group(&self, cx: &App) -> Option<ProjectGroupEntry> {
        self.picker.read(cx).delegate.selected_project_group()
    }

    fn new_thread_in_group(&mut self, _: &NewThreadInGroup, window: &mut Window, cx: &mut Context<Self>) {
        let Some(group) = self.selected_project_group(cx) else {
            return;
        };
        self.show_thread_list(cx);
        self.picker.update(cx, |picker, cx| {
            picker.delegate.open_new_thread_in_group(&group, window, cx);
        });
    }

    fn on_toggle_archive(&mut self, _: &ToggleArchive, window: &mut Window, cx: &mut Context<Self>) {
        self.toggle_archive(window, cx);
    }

    fn focus_sidebar_filter(
        &mut self,
        _: &FocusSidebarFilter,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match &self.view {
            SidebarView::ThreadList => {
                let handle = self.picker.read(cx).focus_handle(cx);
                window.focus(&handle, cx);
            }
            SidebarView::Archive(view) => {
                view.update(cx, |view, cx| view.focus_filter_editor(window, cx));
            }
        }
    }

    fn show_archive(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let active_workspace = self
            .multi_workspace
            .read(cx)
            .workspaces()
            .get(self.multi_workspace.read(cx).active_workspace_index())
            .cloned();
        let Some(active_workspace) = active_workspace else {
            return;
        };

        let archive_view =
            cx.new(|cx| ThreadsArchiveView::new(active_workspace.downgrade(), window, cx));
        let subscription = cx.subscribe_in(
            &archive_view,
            window,
            |this, _, event: &ThreadsArchiveViewEvent, window, cx| match event {
                ThreadsArchiveViewEvent::Close => this.show_thread_list(cx),
                ThreadsArchiveViewEvent::Unarchive { thread } => {
                    this.show_thread_list(cx);
                    this.activate_archived_thread(thread.clone(), window, cx);
                }
            },
        );

        archive_view.update(cx, |view, cx| view.focus_filter_editor(window, cx));
        self._archive_subscription = Some(subscription);
        self.view = SidebarView::Archive(archive_view);
        cx.notify();
    }

    fn show_thread_list(&mut self, cx: &mut Context<Self>) {
        self.view = SidebarView::ThreadList;
        self._archive_subscription = None;
        cx.notify();
    }

    fn workspace_paths_key(workspace: &Entity<Workspace>, cx: &App) -> String {
        let paths: Vec<_> = workspace
            .read(cx)
            .worktrees(cx)
            .filter(|wt| wt.read(cx).is_visible())
            .map(|wt| wt.read(cx).abs_path())
            .collect();
        sorted_paths_key(&paths)
    }

    fn open_archived_thread_in_workspace(
        workspace: &Entity<Workspace>,
        metadata: &ThreadMetadata,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(panel) = workspace.read(cx).panel::<AgentPanel>(cx) else {
            return;
        };

        let mut thread = AgentSessionInfo::new(metadata.session_id.clone());
        thread.cwd = Self::archived_thread_paths(metadata).first().cloned();
        thread.title = Some(metadata.title.clone());
        thread.updated_at = Some(metadata.updated_at);

        let agent = match metadata.agent_id.as_ref() {
            "Zed Agent" => agent_ui::ExternalAgent::NativeAgent,
            GEMINI_NAME => agent_ui::ExternalAgent::Gemini,
            CLAUDE_CODE_NAME => agent_ui::ExternalAgent::ClaudeCode,
            CODEX_NAME => agent_ui::ExternalAgent::Codex,
            name => agent_ui::ExternalAgent::Custom {
                name: name.to_string().into(),
            },
        };

        panel.update(cx, |panel, cx| {
            panel.open_thread_with_agent(agent, thread, window, cx);
        });
    }

    fn archived_thread_paths(metadata: &ThreadMetadata) -> Vec<PathBuf> {
        if metadata.folder_paths.is_empty() {
            metadata
                .main_worktree_paths
                .paths()
                .iter()
                .map(|path| path.to_path_buf())
                .collect()
        } else {
            metadata
                .folder_paths
                .paths()
                .iter()
                .map(|path| path.to_path_buf())
                .collect()
        }
    }

    fn activate_archived_thread(
        &mut self,
        metadata: ThreadMetadata,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let target_paths = Self::archived_thread_paths(&metadata);
        let target_key = sorted_paths_key(&target_paths);
        let maybe_index = self
            .multi_workspace
            .read(cx)
            .workspaces()
            .iter()
            .enumerate()
            .find_map(|(index, workspace)| {
                (Self::workspace_paths_key(workspace, cx) == target_key).then_some(index)
            });

        if let Some(index) = maybe_index {
            let workspace = self.multi_workspace.read(cx).workspaces()[index].clone();
            self.multi_workspace.update(cx, |multi_workspace, cx| {
                multi_workspace.activate_index(index, window, cx);
            });
            Self::open_archived_thread_in_workspace(&workspace, &metadata, window, cx);
            return;
        }

        let paths = target_paths;
        if paths.is_empty() {
            let workspace = self.multi_workspace.read(cx).workspace().clone();
            Self::open_archived_thread_in_workspace(&workspace, &metadata, window, cx);
            return;
        }

        let multi_workspace = self.multi_workspace.clone();
        cx.spawn_in(window, async move |this, cx| {
            let open_task = multi_workspace.update_in(cx, |multi_workspace, window, cx| {
                multi_workspace.open_project(paths, OpenMode::Activate, window, cx)
            })?;
            open_task.await?;

            this.update_in(cx, |this, window, cx| {
                let target_paths = Self::archived_thread_paths(&metadata);
                let target_key = sorted_paths_key(&target_paths);
                let workspace = this
                    .multi_workspace
                    .read(cx)
                    .workspaces()
                    .iter()
                    .find(|workspace| Self::workspace_paths_key(workspace, cx) == target_key)
                    .cloned()
                    .unwrap_or_else(|| this.multi_workspace.read(cx).workspace().clone());
                Self::open_archived_thread_in_workspace(&workspace, &metadata, window, cx);
            })?;

            Ok::<(), db::anyhow::Error>(())
        })
        .detach_and_log_err(cx);
    }
}

impl WorkspaceSidebar for Sidebar {
    fn width(&self, _cx: &App) -> Pixels {
        self.width
    }

    fn set_width(&mut self, width: Option<Pixels>, cx: &mut Context<Self>) {
        self.width = width.unwrap_or(DEFAULT_WIDTH).clamp(MIN_WIDTH, MAX_WIDTH);
        cx.notify();
    }

    fn has_notifications(&self, cx: &App) -> bool {
        !self.picker.read(cx).delegate.notified_groups.is_empty()
    }

    fn serialized_state(&self, cx: &App) -> Option<String> {
        let mut state = self.picker.read(cx).delegate.serialized_state();
        state.width = Some(f32::from(self.width));
        state.active_view = match self.view {
            SidebarView::ThreadList => SerializedSidebarView::ThreadList,
            SidebarView::Archive(_) => SerializedSidebarView::Archive,
        };
        serde_json::to_string(&state).ok()
    }

    fn restore_serialized_state(
        &mut self,
        state: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(serialized) = serde_json::from_str::<SerializedSidebarState>(state).ok() else {
            return;
        };

        if let Some(width) = serialized.width {
            self.width = px(width).clamp(MIN_WIDTH, MAX_WIDTH);
        }

        let project_group_keys: Vec<ProjectGroupKey> = self
            .multi_workspace
            .read(cx)
            .project_groups(cx)
            .map(|(key, _)| key)
            .collect();

        let collapsed_groups: HashSet<ProjectGroupKey> = serialized
            .collapsed_groups
            .iter()
            .filter_map(|paths| {
                let path_key = sorted_paths_key(paths);
                project_group_keys
                    .iter()
                    .find(|key| sorted_paths_key(key.path_list().paths()) == path_key)
                    .cloned()
            })
            .collect();
        let expanded_groups: HashMap<ProjectGroupKey, usize> = serialized
            .expanded_groups
            .iter()
            .filter_map(|(paths, batches)| {
                let path_key = sorted_paths_key(paths);
                project_group_keys
                    .iter()
                    .find(|key| sorted_paths_key(key.path_list().paths()) == path_key)
                    .cloned()
                    .map(|key| (key, *batches))
            })
            .collect();

        self.picker.update(cx, |picker, cx| {
            picker.delegate.collapsed_groups = collapsed_groups;
            picker.delegate.expanded_groups = expanded_groups;
            let query = picker.query(cx);
            picker.update_matches(query, window, cx);
        });
        self.queue_refresh(self.multi_workspace.clone(), window, cx);

        if matches!(serialized.active_view, SerializedSidebarView::Archive) {
            self.show_archive(window, cx);
        } else {
            self.show_thread_list(cx);
        }
    }
}

impl Focusable for Sidebar {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        match &self.view {
            SidebarView::ThreadList => self.picker.read(cx).focus_handle(cx),
            SidebarView::Archive(view) => view.read(cx).focus_handle(cx),
        }
    }
}

fn sorted_paths_key<P: AsRef<Path>>(paths: &[P]) -> String {
    let mut sorted: Vec<String> = paths
        .iter()
        .map(|p| p.as_ref().to_string_lossy().to_string())
        .collect();
    sorted.sort();
    sorted.join("\n")
}

fn format_thread_timestamp(timestamp: chrono::DateTime<chrono::Utc>) -> SharedString {
    let timestamp = OffsetDateTime::from_unix_timestamp(timestamp.timestamp())
        .unwrap_or_else(|_| OffsetDateTime::now_utc());
    time_format::format_localized_timestamp(
        timestamp,
        OffsetDateTime::now_utc(),
        time::UtcOffset::UTC,
        time_format::TimestampFormat::Relative,
    )
    .into()
}

fn read_thread_title_map() -> Option<HashMap<String, String>> {
    let json = KEY_VALUE_STORE
        .read_kvp(LAST_THREAD_TITLES_KEY)
        .log_err()
        .flatten()?;
    serde_json::from_str(&json).log_err()
}

impl Render for Sidebar {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let titlebar_height = ui::utils::platform_title_bar_height(window);
        let ui_font = theme::setup_ui_font(window, cx);
        let is_focused = self.focus_handle(cx).is_focused(window);
        let showing_archive = matches!(self.view, SidebarView::Archive(_));

        let focus_tooltip_label = if is_focused {
            "Focus Workspace"
        } else {
            "Focus Sidebar"
        };

        v_flex()
            .id("workspace-sidebar")
            .key_context("WorkspaceSidebar")
            .font(ui_font)
            .h_full()
            .w(self.width)
            .bg(cx.theme().colors().surface_background)
            .border_r_1()
            .border_color(cx.theme().colors().border)
            .child(
                h_flex()
                    .flex_none()
                    .h(titlebar_height)
                    .w_full()
                    .mt_px()
                    .pb_px()
                    .pr_1()
                    .when(cfg!(target_os = "macos"), |this| {
                        this.pl(px(TRAFFIC_LIGHT_PADDING))
                    })
                    .when(cfg!(not(target_os = "macos")), |this| this.pl_2())
                    .justify_between()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .child({
                        let focus_handle = cx.focus_handle();
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(
                                IconButton::new("close-sidebar", IconName::WorkspaceNavOpen)
                                    .icon_size(IconSize::Small)
                                    .tooltip(Tooltip::element(move |_, cx| {
                                        v_flex()
                                            .gap_1()
                                            .child(
                                                h_flex()
                                                    .gap_2()
                                                    .justify_between()
                                                    .child(Label::new("Close Sidebar"))
                                                    .child(KeyBinding::for_action_in(
                                                        &ToggleWorkspaceSidebar,
                                                        &focus_handle,
                                                        cx,
                                                    )),
                                            )
                                            .child(
                                                h_flex()
                                                    .pt_1()
                                                    .gap_2()
                                                    .border_t_1()
                                                    .border_color(
                                                        cx.theme().colors().border_variant,
                                                    )
                                                    .justify_between()
                                                    .child(Label::new(focus_tooltip_label))
                                                    .child(KeyBinding::for_action_in(
                                                        &FocusWorkspaceSidebar,
                                                        &focus_handle,
                                                        cx,
                                                    )),
                                            )
                                            .into_any_element()
                                    }))
                                    .on_click(cx.listener(|_this, _, _window, cx| {
                                        cx.emit(SidebarEvent::Close);
                                    })),
                            )
                            .child(
                                v_flex()
                                    .gap_0()
                                    .child(Label::new("Agents"))
                                    .child(
                                        Label::new(if showing_archive {
                                            "Archive"
                                        } else {
                                            "Projects"
                                        })
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                    ),
                            )
                    })
                    .child(
                        IconButton::new(
                            if showing_archive {
                                "show-thread-list"
                            } else {
                                "show-archive"
                            },
                            IconName::HistoryRerun,
                        )
                        .icon_size(IconSize::Small)
                        .tooltip(|_window, cx| {
                            Tooltip::for_action("Toggle Archive", &ToggleArchive, cx)
                        })
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.toggle_archive(window, cx);
                        })),
                    )
                    .child(
                        IconButton::new("new-thread-in-selected-group", IconName::Thread)
                            .icon_size(IconSize::Small)
                            .tooltip(|_window, cx| {
                                Tooltip::for_action(
                                    "New Thread In Selected Project",
                                    &NewThreadInGroup,
                                    cx,
                                )
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.new_thread_in_group(&NewThreadInGroup, window, cx);
                            })),
                    )
                    .child(
                        IconButton::new("new-workspace", IconName::Plus)
                            .icon_size(IconSize::Small)
                            .tooltip(|_window, cx| {
                                Tooltip::for_action("New Workspace", &NewWorkspaceInWindow, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.multi_workspace.update(cx, |multi_workspace, cx| {
                                    multi_workspace.create_workspace(window, cx);
                                });
                            })),
                    ),
            )
            .child(match &self.view {
                SidebarView::ThreadList => self.picker.clone().into_any_element(),
                SidebarView::Archive(view) => view.clone().into_any_element(),
            })
            .on_action(cx.listener(Self::new_thread_in_group))
            .on_action(cx.listener(Self::on_toggle_archive))
            .on_action(cx.listener(Self::focus_sidebar_filter))
            .on_action(cx.listener(Self::on_toggle_thread_switcher))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use feature_flags::FeatureFlagAppExt as _;
    use fs::FakeFs;
    use gpui::TestAppContext;
    use settings::SettingsStore;

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            cx.update_flags(false, vec!["agent-v2".into()]);
        });
    }

    fn set_thread_info_and_refresh(
        sidebar: &Entity<Sidebar>,
        multi_workspace: &Entity<MultiWorkspace>,
        index: usize,
        title: &str,
        status: AgentThreadStatus,
        cx: &mut gpui::VisualTestContext,
    ) {
        sidebar.update_in(cx, |s, _window, _cx| {
            s.set_test_thread_info(index, SharedString::from(title.to_string()), status.clone());
        });
        multi_workspace.update_in(cx, |_, _window, cx| cx.notify());
        cx.run_until_parked();
    }

    fn has_notifications(sidebar: &Entity<Sidebar>, cx: &mut gpui::VisualTestContext) -> bool {
        sidebar.read_with(cx, |s, cx| s.has_notifications(cx))
    }

    #[gpui::test]
    async fn test_notification_on_running_to_completed_transition(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn Fs>::set_global(fs.clone(), cx));
        let project = project::Project::test(fs, [], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));

        let sidebar = multi_workspace.update_in(cx, |_mw, window, cx| {
            let mw_handle = cx.entity();
            cx.new(|cx| Sidebar::new(mw_handle, window, cx))
        });
        multi_workspace.update_in(cx, |mw, window, cx| {
            mw.register_sidebar(sidebar.clone(), window, cx);
        });
        cx.run_until_parked();

        // Create a second workspace and switch to it so workspace 0 is background.
        multi_workspace.update_in(cx, |mw, window, cx| {
            mw.create_workspace(window, cx);
        });
        cx.run_until_parked();
        multi_workspace.update_in(cx, |mw, window, cx| {
            mw.activate_index(1, window, cx);
        });
        cx.run_until_parked();

        assert!(
            !has_notifications(&sidebar, cx),
            "should have no notifications initially"
        );

        set_thread_info_and_refresh(
            &sidebar,
            &multi_workspace,
            0,
            "Test Thread",
            AgentThreadStatus::Running,
            cx,
        );

        assert!(
            !has_notifications(&sidebar, cx),
            "Running status alone should not create a notification"
        );

        set_thread_info_and_refresh(
            &sidebar,
            &multi_workspace,
            0,
            "Test Thread",
            AgentThreadStatus::Completed,
            cx,
        );

        assert!(
            has_notifications(&sidebar, cx),
            "Running → Completed transition should create a notification"
        );
    }

    #[gpui::test]
    async fn test_no_notification_for_active_workspace(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn Fs>::set_global(fs.clone(), cx));
        let project = project::Project::test(fs, [], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));

        let sidebar = multi_workspace.update_in(cx, |_mw, window, cx| {
            let mw_handle = cx.entity();
            cx.new(|cx| Sidebar::new(mw_handle, window, cx))
        });
        multi_workspace.update_in(cx, |mw, window, cx| {
            mw.register_sidebar(sidebar.clone(), window, cx);
        });
        cx.run_until_parked();

        // Workspace 0 is the active workspace — thread completes while
        // the user is already looking at it.
        set_thread_info_and_refresh(
            &sidebar,
            &multi_workspace,
            0,
            "Test Thread",
            AgentThreadStatus::Running,
            cx,
        );
        set_thread_info_and_refresh(
            &sidebar,
            &multi_workspace,
            0,
            "Test Thread",
            AgentThreadStatus::Completed,
            cx,
        );

        assert!(
            !has_notifications(&sidebar, cx),
            "should not notify for the workspace the user is already looking at"
        );
    }

    #[gpui::test]
    async fn test_notification_cleared_on_workspace_activation(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn Fs>::set_global(fs.clone(), cx));
        let project = project::Project::test(fs, [], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));

        let sidebar = multi_workspace.update_in(cx, |_mw, window, cx| {
            let mw_handle = cx.entity();
            cx.new(|cx| Sidebar::new(mw_handle, window, cx))
        });
        multi_workspace.update_in(cx, |mw, window, cx| {
            mw.register_sidebar(sidebar.clone(), window, cx);
        });
        cx.run_until_parked();

        // Create a second workspace so we can switch away and back.
        multi_workspace.update_in(cx, |mw, window, cx| {
            mw.create_workspace(window, cx);
        });
        cx.run_until_parked();

        // Switch to workspace 1 so workspace 0 becomes a background workspace.
        multi_workspace.update_in(cx, |mw, window, cx| {
            mw.activate_index(1, window, cx);
        });
        cx.run_until_parked();

        // Thread on workspace 0 transitions Running → Completed while
        // the user is looking at workspace 1.
        set_thread_info_and_refresh(
            &sidebar,
            &multi_workspace,
            0,
            "Test Thread",
            AgentThreadStatus::Running,
            cx,
        );
        set_thread_info_and_refresh(
            &sidebar,
            &multi_workspace,
            0,
            "Test Thread",
            AgentThreadStatus::Completed,
            cx,
        );

        assert!(
            has_notifications(&sidebar, cx),
            "background workspace completion should create a notification"
        );

        // Switching back to workspace 0 should clear the notification.
        multi_workspace.update_in(cx, |mw, window, cx| {
            mw.activate_index(0, window, cx);
        });
        cx.run_until_parked();

        assert!(
            !has_notifications(&sidebar, cx),
            "notification should be cleared when workspace becomes active"
        );
    }
}
