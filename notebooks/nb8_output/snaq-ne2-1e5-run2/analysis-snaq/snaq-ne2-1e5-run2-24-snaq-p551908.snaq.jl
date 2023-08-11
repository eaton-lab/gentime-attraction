
#!/usr/bin/env julia

# check for required packages
using PhyloNetworks
using CSV
using Distributed
using DataFrames

# parallelize
addprocs(4)
@everywhere using PhyloNetworks

# load quartet-CF object from table
d_sp = readTableCF("nb8_output/snaq-ne2-1e5-run2/analysis-snaq/snaq-ne2-1e5-run2-24-snaq-p551908.snaq.CFs.csv")

# load starting network
netin = readTopology("(((r0,r1),(r2,r3)),((r4,r5),(r6,r7)));")

# infer the network
snaq!(netin, d_sp, hmax=1, filename="nb8_output/snaq-ne2-1e5-run2/analysis-snaq/snaq-ne2-1e5-run2-24-snaq-p551908.snaq.net-1", seed=123, runs=8)
