package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.analysis.Optimum;

/**
 * @author David B. Bracewell
 */
public interface LossFunction extends ObjectiveFunction {

   @Override
   default Optimum getOptimum() {
      return Optimum.MINIMUM;
   }

}//END OF LossFunction
