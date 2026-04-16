![Rust](https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust)
![GPUI](https://img.shields.io/badge/GPUI-2F6FED?style=flat-square)
![Agent UI](https://img.shields.io/badge/Agent%20UI-0F9D58?style=flat-square)
![Chat UX](https://img.shields.io/badge/Chat%20UX-FF6F00?style=flat-square)
![Linux Only](https://img.shields.io/badge/Linux%20Only-FCC624?style=flat-square&logo=linux&logoColor=000)

# Personal fork of Zed with custom agent/chat UI

This branch is not just an upstream mirror.

## What's Customized

- agent panel header and navigation behavior
- thread history UI for both ACP and text threads
- better empty and pending thread titles: `New Thread` / `New Thread...`
- keyboard-friendly delete behavior in thread history
- agent-aware ACP history opening, so recent/history entries open with the correct agent context
- smoother ACP streaming behavior in chat replies
- provider modal improvements such as masked API key input
- picker disabling while a thread is actively generating
- selective upstream agent/chat UX ports without pulling the full upstream repo clutter

## What Was Kept

- `crates/`, `assets/`, `extensions/`, `legal/`
- core Cargo workspace files
- linux-local helper scripts still useful for setup/install
- license and third-party notice files

## The Philosophy

- this `clean` branch is for local UI/UX, Linux-only cleanup, and selected upstream ports
- was intentionally stripped of most CI, cloud, workflow, docs, packaging, and repo-maintenance files

## Dependencies

```bash
sudo apt install clang mold pkg-config libx11-dev libxkbcommon-dev libxcb1-dev libxcb-shape0-dev libasound2-dev libpango1.0-dev libgtk-3-dev libwayland-dev libx11-xcb-dev libxkbcommon-x11-dev
```

## Build

```bash
./script/bundle-deb
```

```bash
cargo clean && ./script/bundle-deb
```

```bash
rm -rf target && ./script/bundle-deb
```

## Install

```bash
sudo dpkg -i target/release/zed-linux-x86_64.deb
```

## Uninstall

```bash
sudo dpkg -r zed
```

## Run

```bash
cargo run -p zed
ZED_STATELESS=1 cargo run -p zed
ZED_LOG=debug cargo run -p zed
```

## Attribution

Based on [zed-industries/zed](https://github.com/zed-industries/zed).

## License

See [LICENSE-GPL](LICENSE-GPL), [LICENSE-AGPL](LICENSE-AGPL), [LICENSE-APACHE](LICENSE-APACHE), and [assets/licenses.md](assets/licenses.md).
