use open_ambient::open_ambient_dir;
use spam_csr::gen_bench_mul;

gen_bench_mul!(mul_hash);

fn main() -> anyhow::Result<()> {
    let dir = open_ambient_dir!("matrices")?;
    bench_mul::<false>(dir)?;

    Ok(())
}
