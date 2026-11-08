import argparse
from pathlib import Path
from .runner import run, REPO


def main():
    parser=argparse.ArgumentParser(description='Portable Delta LongMemEval V2 harness')
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--data',type=Path,default=REPO/'data')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget-usd',type=float,required=True,help='Explicit shared API spending cap for this output directory')
    parser.add_argument('--workers',type=int,default=2,help='Concurrent question workers')
    parser.add_argument('--tier',choices=('small','medium'),default='small')
    args=parser.parse_args()
    run(args.data,args.output,args.budget_usd,args.workers,prepare_only=args.command=='prepare',tier=args.tier)

if __name__=='__main__': main()
