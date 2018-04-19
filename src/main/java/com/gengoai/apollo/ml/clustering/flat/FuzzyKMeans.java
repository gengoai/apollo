package com.gengoai.apollo.ml.clustering.flat;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.clustering.ApacheClusterable;
import com.gengoai.apollo.ml.clustering.ApacheDistanceMeasure;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.apollo.stat.measure.DistanceMeasure;
import com.gengoai.mango.stream.MStream;
import com.gengoai.apollo.ml.clustering.ApacheClusterable;
import com.gengoai.apollo.ml.clustering.ApacheDistanceMeasure;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.math3.ml.clustering.FuzzyKMeansClusterer;
import org.apache.commons.math3.random.Well19937c;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Clusters using Apache Math's implementation of the <a href="https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/fk_means.htm">Fuzzy
 * K-Means</a> algorithm.
 *
 * @author David B. Bracewell
 */
public class FuzzyKMeans extends Clusterer<FlatClustering> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int K;
   @Getter
   @Setter
   private int maxIterations = -1;
   @Getter
   @Setter
   private double fuzziness;
   @Getter
   @Setter
   private double epsilon = 1e-3;
   @Getter
   @Setter(onParam = @_({@NonNull}))
   private DistanceMeasure distanceMeasure = Distance.Euclidean;

   private FuzzyKMeans() {
      this(2, 2.0);
   }

   /**
    * Instantiates a new Fuzzy K-means clusterer.
    *
    * @param k         the number of clusters
    * @param fuzziness the fuzziness factor, must be > 1.0
    */
   public FuzzyKMeans(int k, double fuzziness) {
      this.fuzziness = fuzziness;
      this.K = k;
   }

   /**
    * Instantiates a new Fuzzy K-means clusterer.
    *
    * @param distanceMeasure the distance measure to use (default Euclidean)
    * @param K               the number of clusters
    * @param maxIterations   the maximum number of iterations to run the algorithm
    * @param fuzziness       the fuzziness factor, must be > 1.0
    * @param epsilon         the convergence criteria (default is 1e-3)
    */
   public FuzzyKMeans(@NonNull DistanceMeasure distanceMeasure, int K, int maxIterations, double fuzziness, double epsilon) {
      this.distanceMeasure = distanceMeasure;
      this.K = K;
      this.maxIterations = maxIterations;
      this.fuzziness = fuzziness;
      this.epsilon = epsilon;
   }

   @Override
   public FlatClustering cluster(MStream<NDArray> instances) {
      FuzzyKMeansClusterer<ApacheClusterable> clusterer = new FuzzyKMeansClusterer<>(this.K,
                                                                                     this.fuzziness,
                                                                                     this.maxIterations,
                                                                                     new ApacheDistanceMeasure(this.distanceMeasure),
                                                                                     this.epsilon,
                                                                                     new Well19937c()
      );
      List<Cluster> clusters = clusterer.cluster(instances.map(ApacheClusterable::new).collect())
                                        .stream()
                                        .filter(c -> c.getPoints().size() > 0)
                                        .map(c -> {
                                           Cluster cp = new Cluster();
                                           cp.setCentroid(NDArrayFactory.wrap(c.getCenter().getPoint()));
                                           c.getPoints().forEach(ap -> cp.addPoint(ap.getVector()));
                                           return cp;
                                        }).collect(Collectors.toList());
      return new FlatCentroidClustering(this, distanceMeasure, clusters);
   }


}// END OF FuzzyKMeans
