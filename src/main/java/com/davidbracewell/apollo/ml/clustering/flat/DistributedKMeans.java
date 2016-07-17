package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.stream.StreamingContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class DistributedKMeans extends Clusterer<FlatHardClustering> {
  private static final long serialVersionUID = 1L;
  private int K = 2;
  private int maxIterations = 100;

  public DistributedKMeans() {
  }

  public DistributedKMeans(int K, int maxIterations) {
    this.K = K;
    this.maxIterations = maxIterations;
  }

  @Override
  public FlatHardClustering cluster(List<LabeledVector> instances) {
    JavaRDD<Vector> rdd = StreamingContext.distributed().stream(instances)
      .map(v -> (Vector) new DenseVector(v.toArray()))
      .asRDD();
    KMeansModel model = org.apache.spark.mllib.clustering.KMeans.train(
      rdd.rdd(),
      K,
      maxIterations
    );
    Map<Long, Iterable<Integer>> assignments = model.predict(rdd)
      .zipWithIndex()
      .mapToPair(t -> new Tuple2<>(t._2(), t._1()))
      .groupByKey()
      .collectAsMap();
    Vector[] centroids = model.clusterCenters();
    List<Cluster> clusters = new ArrayList<>();
    for (int i = 0; i < centroids.length; i++) {
      Cluster cluster = new Cluster();
      cluster.setCentroid(new com.davidbracewell.apollo.linalg.DenseVector(centroids[i].toArray()));
      assignments.get(i).forEach(index -> cluster.addPoint(instances.get(index)));
      clusters.add(cluster);
    }
    FlatHardClustering fc = new FlatHardClustering(getEncoderPair(), Distance.Euclidean, clusters);
    return fc;
  }


  /**
   * Gets k.
   *
   * @return the k
   */

  public int getK() {
    return K;
  }

  /**
   * Sets k.
   *
   * @param k the k
   */
  public void setK(int k) {
    K = k;
  }

  /**
   * Gets max iterations.
   *
   * @return the max iterations
   */
  public int getMaxIterations() {
    return maxIterations;
  }

  /**
   * Sets max iterations.
   *
   * @param maxIterations the max iterations
   */
  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }


}// END OF DistributedKMeans
