package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.analysis.Optimum;

/**
 * @author David B. Bracewell
 */
public interface ObjectiveFunction {

   Optimum getOptimum();

   double calculate(double p, double y);

}//END OF ObjectiveFunction
