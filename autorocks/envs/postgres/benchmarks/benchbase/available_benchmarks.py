from sysgym.utils.enum import BenchmarkTask


class BenchmarkClass(BenchmarkTask):
    TPC_C = "tpcc"
    TPC_H = "tpch"
    # TAT_P = "tatp"
    WIKI = "wikipedia"
    # RESOURCE_STRESSER = "resourcestresser"
    # TWITTER = "twitter"
    # EPINIONS = "epinions"
    YCSB = "ycsb"
    # SEATS = "seats"
    # AUCTION = "auctionmark"
    # CH_BENCHMARK = "chbenchmark"
    # VOTER = "voter"
    # SI_BENCH = "sibench"
    NOOP = "noop"
    # SMALL_BANK = "smallbank"
    # HY_ADAPT = "hyadapt"
