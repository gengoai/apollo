package com.gengoai.apollo.ml.pgm;

import com.gengoai.collection.counter.Counter;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public interface Topic extends Serializable {


   /**
    * Gets the feature and their probabilities for a given topic
    *
    * @return the feature distribution
    */

   Counter<String> featureDistribution();


}//END OF Topic
