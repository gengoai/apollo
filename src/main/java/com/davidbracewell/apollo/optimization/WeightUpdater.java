package com.davidbracewell.apollo.optimization;

/**
 * @author David B. Bracewell
 */
public interface WeightUpdater {

   void reset();

   void update(Weights weights, Weights gradient);

}//END OF WeightUpdater
