"""Explicit configuration injection into the frozen rate9 execution engine."""
import argparse
from . import protocol as P


def bind(variant,screen):
    import pressure9.runner as runner
    from pressure9.policies import set_progress
    runner.configuration=lambda method,seed=87021,n=150:P.configuration(method,seed,n,variant,screen)
    runner.source_hashes=P.source_hashes
    runner.VALIDATION_SEEDS=P.VALIDATION_SEEDS
    runner.set_progress=lambda p,step,total:set_progress(p,step,1208)
    return runner


def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('train','evaluate'))
    a.add_argument('--out',required=True);a.add_argument('--method',required=True)
    a.add_argument('--seed',type=int,required=True);a.add_argument('--variant',default='control',choices=tuple(P.VARIANTS))
    a.add_argument('--screen',action='store_true');a.add_argument('--resume',action='store_true')
    a.add_argument('--maximum-seconds',type=float,default=9000);a.add_argument('--weights')
    args=a.parse_args();runner=bind(args.variant,args.screen)
    if args.mode=='train':
        done=runner.train(args.out,args.method,args.seed,args.resume,args.maximum_seconds)
    else:
        import pressure9.evaluate as evaluation
        evaluation.configuration=lambda method,seed=87021,n=150:P.configuration(method,seed,n,args.variant,False)
        evaluation.TEST_SEEDS=P.TEST_SEEDS
        done=evaluation.evaluate(args.out,args.method,args.seed,args.weights,args.maximum_seconds)
    raise SystemExit(0 if done else 2)


if __name__=='__main__':main()
