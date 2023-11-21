real gev_cdf(real y, real mu, real sigma, real xi) {
  
  if (abs(xi) < 1e-10) {
    real z = (y - mu) / sigma;
    return exp(-exp(z));
  } else {
    real z = 1 + xi * (y - mu) / sigma;
    if (z > 0) {
      return exp(-pow(z, -1/xi));
    } else {
      reject("Found incompatible GEV parameter values");
    }
  }
}

real gev_lpdf(real y, real mu, real sigma, real xi) {
  if (abs(xi) < 1e-10) {
    real z = (y - mu) / sigma;
    return -log(sigma) - z - exp(-z);
  } else {
    real z = 1 + xi * (y - mu) / sigma;
    if (z > 0) {
      return -log(sigma) - (1 + 1/xi) * log(z) - pow(z, -1/xi);
    } else {
      reject("Found incompatible GEV parameter values");
    }
  }
}

