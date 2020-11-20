# Bayesian Regression using the Hill Equation 

Simple `pytorch/pyro` implementation of the hill equation in a bayesian regression framework. Allows for placement of priors on variables as well as propagation of uncertainty onto the summary metrics like IC50. 

We use the model: 
$$ y = E_0 + (\frac{ E_{max} - E_0 } {1 + (\frac{EC_{50}}{x}})^H) $$

Where, 
> **y**        : cell viability [0,1]  
> **x**        : log10 concentration  
> **$E_0$**    : minimum inhibition (max cell viab.)  
> **$E_{max}$**: maximum inhibition (min cell viab.)   
> **$EC_{50}$**: concetration at which there is 50% maximum inhibition (e.g., conc (x) at which y = ($E_0$ - $E_{max}$)/2 + $E_{max}$)  
> **H**        : Hill coefficient  


We believe this method is especially valuable in low data scenarios, or when there is poor concordance between replicates. 

See the [tutorial](./tutorial.ipynb) for example on how to use this code.

Email `evansna@ohsu.edu` for questions. 
