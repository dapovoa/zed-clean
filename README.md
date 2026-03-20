![Rust](https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust)
![GPUI](https://img.shields.io/badge/GPUI-2F6FED?style=flat-square)
![Agent UI](https://img.shields.io/badge/Agent%20UI-0F9D58?style=flat-square)
![Chat UX](https://img.shields.io/badge/Chat%20UX-FF6F00?style=flat-square)
![Linux Only](https://img.shields.io/badge/Linux%20Only-FCC624?style=flat-square&logo=linux&logoColor=000)

# Personal Linux-first fork of Zed with custom agent/chat UI

This branch is not just an upstream mirror. It is the branch where local product decisions live.

## What Is Customized Here

- agent panel header and navigation behavior
- thread history UI for both ACP and text threads
- better empty and pending thread titles: `New Thread` / `New Thread...`
- keyboard-friendly delete behavior in thread history
- agent-aware ACP history opening, so recent/history entries open with the correct agent context
- smoother ACP streaming behavior in chat replies
- provider modal improvements such as masked API key input
- picker disabling while a thread is actively generating
- selective upstream agent/chat UX ports without pulling the full upstream repo clutter

## Repo Philosophy

- `main` should stay close to upstream `zed-industries/zed`
- `clean` is the working branch for local UI, UX, Linux-only cleanup, and selected upstream ports
- the repository is intentionally stripped of most CI, cloud, workflow, docs, packaging, and repo-maintenance files that are not needed for this branch

## Run

```bash
cargo run -p zed
```

## Kept On Purpose

- `crates/`, `assets/`, `extensions/`, `legal/`
- core Cargo workspace files
- Linux-local helper scripts still useful for setup/install
- license and third-party notice files

## Attribution

Based on [zed-industries/zed](https://github.com/zed-industries/zed).

## License

See [LICENSE-GPL](LICENSE-GPL), [LICENSE-AGPL](LICENSE-AGPL), [LICENSE-APACHE](LICENSE-APACHE), and [assets/licenses.md](assets/licenses.md).
