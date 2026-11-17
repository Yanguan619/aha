use clap::Parser;

use crate::cli::{
    args::{Cli, Commands},
    run_cli, run_delete, run_download, run_list, run_multi, run_probe, run_ps, run_run, run_serv,
};

mod cli;
#[allow(unused)]
mod params;
mod server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli::init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Commands::Cli(args) => run_cli(args).await,
        Commands::Serv(args) => run_serv(args).await,
        Commands::Probe(args) => run_probe(args).await,
        #[cfg(target_os = "android")]
        Commands::Npu(args) => crate::cli::run_npu(args),
        #[cfg(feature = "qnn")]
        Commands::Qnn(args) => crate::cli::run_qnn(args),
        Commands::Multi(args) => run_multi(args).await,
        Commands::Ps(args) => run_ps(args),
        Commands::Delete(args) => run_delete(args),
        Commands::Download(args) => run_download(args).await,
        Commands::Run(args) => run_run(args),
        Commands::List(args) => run_list(args),
    }
}
