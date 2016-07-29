package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.math3.ml.clustering.FuzzyKMeansClusterer;
import org.apache.commons.math3.random.JDKRandomGenerator;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class FuzzyKMeans extends Clusterer<FlatClustering> {
  private static final long serialVersionUID = 1L;
  @Getter
  @Setter
  private int K;
  @Getter
  @Setter
  private int maxIterations;
  @Getter
  @Setter
  private double fuzziness;
  @Getter
  @Setter
  private double epsilon;
  @Getter
  @Setter(onParam = @_({@NonNull}))
  private DistanceMeasure distanceMeasure;

  public FuzzyKMeans() {
    this(Distance.Euclidean, 2);
  }

  public FuzzyKMeans(DistanceMeasure distanceMeasure, int K) {
    this(distanceMeasure, K, 1.1);
  }

  public FuzzyKMeans(DistanceMeasure distanceMeasure, int K, double fuzziness) {
    this(distanceMeasure, K, 100, fuzziness, 1e-3);
  }

  public FuzzyKMeans(DistanceMeasure distanceMeasure, int K, int maxIterations, double fuzziness) {
    this(distanceMeasure, K, maxIterations, fuzziness, 1e-3);
  }

  public FuzzyKMeans(@NonNull DistanceMeasure distanceMeasure, int K, int maxIterations, double fuzziness, double epsilon) {
    this.distanceMeasure = distanceMeasure;
    this.K = K;
    this.maxIterations = maxIterations;
    this.fuzziness = fuzziness;
    this.epsilon = epsilon;
  }

  @Override
  public FlatClustering cluster(MStream<Vector> instances) {
    FuzzyKMeansClusterer<ApacheClusterable> clusterer = new FuzzyKMeansClusterer<>(
      this.K,
      this.fuzziness,
      this.maxIterations,
      new ApacheDistanceMeasure(this.distanceMeasure),
      this.epsilon,
      new JDKRandomGenerator()
    );
    List<Cluster> clusters = clusterer.cluster(instances.map(ApacheClusterable::new).collect())
      .stream()
      .filter(c -> c.getPoints().size() > 0)
      .map(c -> {
        Cluster cp = new Cluster();
        cp.setCentroid(new DenseVector(c.getCenter().getPoint()));
        c.getPoints().forEach(ap -> cp.addPoint(ap.getVector()));
        return cp;
      }).collect(Collectors.toList());
    return new KMeansClustering(getEncoderPair(), distanceMeasure, clusters);
  }


}// END OF FuzzyKMeans
