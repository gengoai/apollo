package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.analysis.Optimum;
import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public interface Optimizer {

   Vector optimize();

   Optimum getGoal();

}//END OF Optimizer
