from nltk import translate
import sys


def bleu_score(hyp, ref):
    #hyp_f = [x.split() for x in open(hyp).readlines()]
    #ref_f = [x.split() for x in open(ref).readlines()]
    hyp_f = open(hyp).readlines()
    ref_f = open(ref).readlines()
    #print hyp_f
    #print ref_f
    print translate.bleu_score(hyp_f, ref_f)


def main(args):

    hypothesis = args[1]
    reference = args[2]
    bleu_score(hypothesis, reference)


if __name__=='__main__':
    sys.exit(main(sys.argv))
