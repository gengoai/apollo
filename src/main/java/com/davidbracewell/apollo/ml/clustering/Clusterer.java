package com.davidbracewell.apollo.ml.clustering;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Learner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.Setter;

/**
 * <p>Base class for clusterer learners.</p>
 *
 * @param <T> the clustering type parameter
 * @author David B. Bracewell
 */
public abstract class Clusterer<T extends Clustering> extends Learner<Instance, T> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private EncoderPair encoderPair;

   /**
    * Clusters a stream of vectors.
    *
    * @param instances the instances
    * @return the clustering model
    */
   public abstract T cluster(MStream<Vector> instances);

   @Override
   public void resetLearnerParameters() {
      this.encoderPair = null;
   }

   @Override
   public T train(Dataset<Instance> dataset) {
      return Cast.as(super.train(dataset));
   }

   protected T trainImpl(Dataset<Instance> dataset) {
      this.encoderPair = dataset.getEncoderPair();
      return cluster(dataset.asVectors());
   }


}// END OF Clusterer
