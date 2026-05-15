fn main() {
    // OneLiner emits absolute IREE object paths with #[link(..., +verbatim)].
    // Adding / lets linkers resolve those paths from Rust's -l: namespec.
    println!("cargo:rustc-link-search=native=/");
}
