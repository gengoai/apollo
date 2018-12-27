package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.ml.Evaluation;

/**
 * @author David B. Bracewell
 */
public interface ClusteringEvaluation extends Evaluation {

   void evaluate(Clustering clustering);

}//END OF ClusteringEvaluation
