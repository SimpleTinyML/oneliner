fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_IREE_RUNTIME").is_some() {
        println!("cargo:rustc-link-search=native=/");
    }
}