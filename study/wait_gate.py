"""Conservative causal distinct-user fill and a separately checked cost gate."""
import math


def fill_probability(population,occupied,needed,events,exposure_s,horizon_s,z=1.645):
    if not 0<=occupied<=population or exposure_s<=0 or horizon_s<0 or events<0:
        raise ValueError('Invalid public history')
    available=population-occupied
    if needed<=0:return 1.
    if needed>available:return 0.
    # Lower confidence rate, not the known simulator future intensity.
    lower=max(0.,events-z*math.sqrt(events))/(population*exposure_s)
    q=-math.expm1(-lower*horizon_s)
    return min(1.,max(0.,sum(math.comb(available,k)*q**k*(1-q)**(available-k)
                            for k in range(needed,available+1))))


def positive_gain(current,future,margin=.002,z=1.645):
    if len(current)!=len(future) or len(current)<2:
        raise ValueError('Paired public model samples required')
    d=[a-b for a,b in zip(current,future)]
    mean=sum(d)/len(d)
    variance=sum((v-mean)**2 for v in d)/(len(d)-1)
    lower=mean-z*math.sqrt(variance/len(d))-margin
    return lower>0,mean,lower
