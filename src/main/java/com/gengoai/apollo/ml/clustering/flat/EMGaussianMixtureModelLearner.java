package com.gengoai.apollo.ml.clustering.flat;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.mango.stream.MStream;
import com.gengoai.apollo.ml.clustering.Clusterer;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization;
import org.apache.commons.math3.util.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class EMGaussianMixtureModelLearner extends Clusterer<GMM> {
   @Getter
   @Setter
   private int K = 100;

   @Override
   public GMM cluster(@NonNull MStream<NDArray> instances) {
      List<NDArray> vectors = instances.collect();
      int numberOfFeatures = getEncoderPair().getFeatureEncoder().size();
      int numberOfDataPoints = vectors.size();
      double[][] data = new double[numberOfDataPoints][numberOfFeatures];
      for (int i = 0; i < numberOfDataPoints; i++) {
         data[i] = vectors.get(i).toArray();
      }
      GMM gmm = new GMM(this, Distance.Euclidean);
      gmm.components = new ArrayList<>(MultivariateNormalMixtureExpectationMaximization.estimate(data, K)
                                                                                       .getComponents().stream()
                                                                                       .map(Pair::getSecond)
                                                                                       .collect(Collectors.toList()));
      return gmm;
   }

}// END OF GMM
