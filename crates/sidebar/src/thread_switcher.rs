use agent_client_protocol as acp;
use gpui::{
    Action as _, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, Modifiers,
    ModifiersChangedEvent, Render, SharedString, prelude::*,
};
use ui::{ThreadItem, prelude::*};
use workspace::{ModalView, Workspace};

use crate::AgentThreadStatus;

#[derive(Clone)]
pub(crate) struct ThreadSwitcherEntry {
    pub session_id: acp::SessionId,
    pub title: SharedString,
    pub icon: IconName,
    pub status: AgentThreadStatus,
    pub workspace: Entity<Workspace>,
    pub project_name: SharedString,
    pub worktree_label: SharedString,
    pub generating_title: bool,
    pub notified: bool,
    pub timestamp: SharedString,
}

pub(crate) enum ThreadSwitcherEvent {
    Preview {
        session_id: acp::SessionId,
        workspace: Entity<Workspace>,
    },
    Confirmed {
        session_id: acp::SessionId,
        workspace: Entity<Workspace>,
    },
    Dismissed,
}

pub(crate) struct ThreadSwitcher {
    focus_handle: FocusHandle,
    entries: Vec<ThreadSwitcherEntry>,
    selected_index: usize,
    init_modifiers: Option<Modifiers>,
}

impl ThreadSwitcher {
    pub fn new(
        entries: Vec<ThreadSwitcherEntry>,
        select_last: bool,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let init_modifiers = window.modifiers().modified().then_some(window.modifiers());
        let selected_index = if entries.is_empty() {
            0
        } else if select_last {
            entries.len() - 1
        } else {
            1.min(entries.len().saturating_sub(1))
        };

        if let Some(entry) = entries.get(selected_index) {
            cx.emit(ThreadSwitcherEvent::Preview {
                session_id: entry.session_id.clone(),
                workspace: entry.workspace.clone(),
            });
        }

        let focus_handle = cx.focus_handle();
        cx.on_focus_out(&focus_handle, window, |_this, _event, _window, cx| {
            cx.emit(ThreadSwitcherEvent::Dismissed);
            cx.emit(DismissEvent);
        })
        .detach();

        Self {
            focus_handle,
            entries,
            selected_index,
            init_modifiers,
        }
    }

    pub fn cycle_selection(&mut self, cx: &mut Context<Self>) {
        if self.entries.is_empty() {
            return;
        }
        self.selected_index = (self.selected_index + 1) % self.entries.len();
        self.emit_preview(cx);
    }

    fn emit_preview(&mut self, cx: &mut Context<Self>) {
        if let Some(entry) = self.entries.get(self.selected_index) {
            cx.emit(ThreadSwitcherEvent::Preview {
                session_id: entry.session_id.clone(),
                workspace: entry.workspace.clone(),
            });
        }
    }

    fn confirm(&mut self, _: &menu::Confirm, _window: &mut gpui::Window, cx: &mut Context<Self>) {
        if let Some(entry) = self.entries.get(self.selected_index) {
            cx.emit(ThreadSwitcherEvent::Confirmed {
                session_id: entry.session_id.clone(),
                workspace: entry.workspace.clone(),
            });
        }
        cx.emit(DismissEvent);
    }

    fn cancel(&mut self, _: &menu::Cancel, _window: &mut gpui::Window, cx: &mut Context<Self>) {
        cx.emit(ThreadSwitcherEvent::Dismissed);
        cx.emit(DismissEvent);
    }

    fn toggle(
        &mut self,
        _: &crate::ToggleThreadSwitcher,
        _window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        self.cycle_selection(cx);
    }

    fn handle_modifiers_changed(
        &mut self,
        event: &ModifiersChangedEvent,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        let Some(init_modifiers) = self.init_modifiers else {
            return;
        };
        if !event.modified() || !init_modifiers.is_subset_of(event) {
            self.init_modifiers = None;
            if self.entries.is_empty() {
                cx.emit(DismissEvent);
            } else {
                window.dispatch_action(menu::Confirm.boxed_clone(), cx);
            }
        }
    }
}

impl ModalView for ThreadSwitcher {}
impl EventEmitter<DismissEvent> for ThreadSwitcher {}
impl EventEmitter<ThreadSwitcherEvent> for ThreadSwitcher {}

impl Focusable for ThreadSwitcher {
    fn focus_handle(&self, _cx: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ThreadSwitcher {
    fn render(&mut self, _window: &mut gpui::Window, cx: &mut Context<Self>) -> impl IntoElement {
        let selected_index = self.selected_index;

        v_flex()
            .key_context("ThreadSwitcher")
            .w(rems(28.))
            .p_1p5()
            .gap_0p5()
            .elevation_3(cx)
            .on_modifiers_changed(cx.listener(Self::handle_modifiers_changed))
            .on_action(cx.listener(Self::confirm))
            .on_action(cx.listener(Self::cancel))
            .on_action(cx.listener(Self::toggle))
            .children(self.entries.iter().enumerate().map(|(ix, entry)| {
                let id = SharedString::from(format!("thread-switcher-{}", entry.session_id));
                ThreadItem::new(id, entry.title.clone())
                    .icon(entry.icon)
                    .running(entry.status == AgentThreadStatus::Running)
                    .generating_title(entry.generating_title)
                    .selected(ix == selected_index)
                    .worktree(if entry.worktree_label.is_empty() {
                        entry.project_name.clone()
                    } else {
                        format!("{} · {}", entry.project_name, entry.worktree_label).into()
                    })
                    .timestamp(entry.timestamp.clone())
                    .generation_done(entry.notified)
                    .into_any_element()
            }))
    }
}
