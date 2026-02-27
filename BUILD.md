# Build

## Dependencies (once)

```sh
script/linux
```

## .deb

```sh
script/bundle-deb
sudo dpkg -i target/release/zed-linux-x86_64.deb
```

Uninstall:

```sh
sudo dpkg -r zed
```

## Direct install (no .deb)

```sh
./script/install-linux
```

Uninstall:

```sh
rm ~/.local/bin/zed
rm ~/.local/libexec/zed-editor
rm -rf ~/.local/share/applications/zed*.desktop
rm -rf ~/.local/share/icons/hicolor/*/apps/zed.png
```

## Quick test (no install)

```sh
cargo run --release --package zed
```

## Clean build artifacts

The `target/` folder can grow to ~95G. To free disk space:

```sh
cargo clean
```

Or delete directly (faster):

```sh
rm -rf target/
```

The project can be rebuilt from scratch afterwards.
