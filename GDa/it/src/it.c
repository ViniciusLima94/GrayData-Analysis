#include <math.h> 

float _log2(float x) {
    if(x==0) {
        return 0;
    }
    else {
        return log2(x);
    }
}

float entropyfromprobabilities(float *p, int n_bins) {
    /* ##################################################################################################### */
    /*  Description: Computes the entropy given a discrete probability distribution. */
    /*  > Inputs: */
    /*  p: A discrete probability distribution. */
    /*  n_bins: Number of bins in the discrete prob. distribution. */
    /*  > Outputs: */
    /*  H: The entropy of the distribution in bits. */
    /* ##################################################################################################### */
    int i   = 0;
    float H = 0;

    for(i=0; i<n_bins; i++) {
        H = H - p[i]*_log2(p[i]);
    }

    return H;
}

float BinEntropy(int *x, int n_bins) {
    /* ##################################################################################################### */
    /*  Description: Computes the entropy of a binary discrete time seres. */
    /*  > Inputs: */
    /*  x: Binary discrete time series (series of 0s and 1s), should be size [N_variables, N_observations]. */
    /*  n_bins: Number of bins in the discrete prob. distribution. */
    /*  > Outputs: */
    /*  H: Entropy of x. */
    /* ##################################################################################################### */
    int i      = 0;
    float p[2] = {0,0};

    // Computing probability of x=1.
    for(i=0; i<n_bins; i++) {
        if(x[i]==1) {
            p[0] = p[0] + 1;
        }
        else {
            continue;
        }
    }
    // Normalizing probability.
    p[0] = p[0]/n_bins;
    p[1] = 1 - p[0];
    // Computing entropy
    return entropyfromprobabilities(p,2);
}

float BinJointEntropy(int *x, int *y, int n_bins) {
    /* ##################################################################################################### */
    /*  Description: Computes the entropy and joint entropy from binary discrete time series. */
    /*  Inputs: */
    /*  x: Binary discrete time series. */
    /*  y: Binary discrete time series. */
    /*  n_bins: Number of bins in the discrete prob. distribution. */
    /*  Outputs: */
    /*  Hxy: The joint entropy between x and y. */
    /* ##################################################################################################### */
    int i     = 0;
    int j     = 0;
    float Hxy = 0;
    float pxy[2][2] = { {0, 0}, {0,0} };

    for(i = 0; i < n_bins; i++) {
        if(x[i]==1 || y[i]==1) {
            pxy[x[i]][y[i]]++;
        }   
    }

    /* Computing probabilities for zeros or a pair of zeros happen */
    pxy[0][0] = n_bins - (pxy[0][1] + pxy[1][0] + pxy[1][1]);

    /* Normalizing probabilities */
    pxy[0][0] = pxy[0][0]/n_bins; 
    pxy[0][1] = pxy[0][1]/n_bins;
    pxy[1][0] = pxy[1][0]/n_bins;
    pxy[1][1] = pxy[1][1]/n_bins;

    /* Computing the joint entropy */
    for(i=0; i<2; i++) {
        for(j=0; j<2; j++) {
            Hxy = Hxy - pxy[i][j]*_log2(pxy[i][j]);
        }
    }

    return Hxy;
}

float BinMutualInformation(int *x, int *y, int n_bins) {

    /* Computing entropies */
    float Hx  = BinEntropy(x, n_bins);
    float Hy  = BinEntropy(y, n_bins);
    float Hxy = BinJointEntropy(x, y, n_bins);
    /* Compute mutual information */
    float MI = Hx + Hy - Hxy;
    return MI;

}
    

