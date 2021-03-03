#include <math.h> 

float _log2(float x) {
    if(x==0) {
        return 0;
    }
    else {
        return log2(x);
    }
}

float st_mi(int *x, int *y, int n_bins) {
   
  int i           = 0;  
  int j           = 0;  
  float px[2]     = { 0, 0 };
  float py[2]     = { 0, 0 };
  float pxy[2][2] = { {0, 0}, {0,0} };

  /* Entropies */
  float Hx  = 0;
  float Hy  = 0;
  float Hxy = 0;  

  for(i = 0; i < n_bins; i++) {

      if(x[i]==1) {
          px[1]++;
      }
      if(y[i]==1) {
          py[1]++;
      }
      if(x[i]==1 || y[i]==1) {
          pxy[x[i]][y[i]]++;
      }
  }

  /* Computing probabilities for zeros or a pair of zeros happen */
  px[0]     = n_bins - px[1];
  py[0]     = n_bins - py[1];
  pxy[0][0] = n_bins - (pxy[0][1] + pxy[1][0] + pxy[1][1]);

  /* Normalizing probabilities */
  px[0]     = px[0]/n_bins; px[1] = px[1]/n_bins;
  py[0]     = py[0]/n_bins; py[1] = py[1]/n_bins;
  pxy[0][0] = pxy[0][0]/n_bins; pxy[0][1] = pxy[0][1]/n_bins;
  pxy[1][0] = pxy[1][0]/n_bins; pxy[1][1] = pxy[1][1]/n_bins;

  /* Computing mutual information */
  for(i=0; i<2; i++) {
      Hx = Hx - px[i]*_log2(px[i]);
      Hy = Hy - py[i]*_log2(py[i]);
  }
  
  for(i=0; i<2; i++) {
    for(j=0; j<2; j++) {
        Hxy = Hxy - pxy[i][j]*_log2(pxy[i][j]);
        }
  }

  float MI =  Hx + Hy - Hxy; 
  return MI;
}
