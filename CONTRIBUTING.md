# Contributing to BloomCraft

Bug reports, API feedback, and pull requests are welcome. This document covers
everything you need to get a contribution merged cleanly.

---

## Reporting Issues

Open an issue at [github.com/ZaudRehman/BloomCraft/issues](https://github.com/ZaudRehman/BloomCraft/issues).

Use the appropriate label:

| Label | Use for |
|---|---|
| `bug` | Incorrect behavior, wrong FPR, panics in safe code |
| `enhancement` | New filter variant, API addition, performance improvement |
| `question` | Usage questions, design rationale, architecture clarification |
| `documentation` | Missing, incorrect, or unclear docs |

For bugs, include:
- Rust version (`rustc --version`)
- BloomCraft version and active feature flags
- A minimal reproduction (ideally a failing `#[test]`)
- The observed vs. expected behavior

---

## Pull Requests

Target the `main` branch for all PRs. Before opening one:

1. **Run the full test suite locally:**
   ```bash
   cargo test --all-features
   ```

2. **Run clippy — zero warnings expected:**
   ```bash
   cargo clippy --all-features -- -D warnings
   ```

3. **Run the benchmarks if your change touches a hot path:**
   ```bash
   cargo bench
   ```

### Checklist

Every PR must satisfy all of the following before it will be reviewed:

- [ ] All existing tests pass (`cargo test --all-features`)
- [ ] New public API items have `///` doc comments with at least one runnable example
- [ ] New behavior has unit tests covering the happy path and at least one edge case
- [ ] An entry has been added to `CHANGELOG.md` under `[Unreleased]`
- [ ] No new `clippy` warnings are introduced

### Adding a New Filter Variant

New filter types carry a higher bar:

- Implement `BloomFilter<T>` fully — no methods may be left as stubs or delegations that silently drop data
- Provide an integration test in `tests/` in addition to unit tests in the module
- Add the variant to the filter selection table and filter variant section in `README.md`
- Add it to the `test_all_filters_accessible` test in `filters/mod.rs`
- Confirm it implements `Send + Sync` and add it to `test_filters_are_send_sync`

### Unsafe Code Policy

Any new `unsafe` block must include a `// SAFETY:` comment that explains:

1. What invariant makes the operation sound
2. Why that invariant holds at this call site
3. What would break if a caller violated it

PRs that introduce `unsafe` without this justification will not be merged,
regardless of whether the code appears correct.

---

## MSRV Policy

BloomCraft targets **Rust 1.70**. Do not use features stabilized after that
version without a corresponding update to:

- The badge in `README.md`
- The `rust-version` field in `Cargo.toml`
- The CI matrix in `.github/workflows/`

---

## Commit Style

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat(filters): add XorBloomFilter variant
fix(scalable): correct fill-rate calculation under Bounded growth
docs(readme): add AtomicPartitionedBloomFilter example
perf(striped): replace modulo with Lemire range reduction
test(counting): add overflow boundary test for 4-bit counter
```

Scope is the module name (`filters`, `sync`, `builder`, `hash`, `metrics`, `serde`).

---

## License

By contributing, you agree that your contributions will be dual-licensed under
the same [MIT OR Apache-2.0](LICENSE) terms as the rest of the project.
