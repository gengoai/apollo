package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.SparkStream;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class DistributedKMeans extends Clusterer<FlatClustering> {
  private static final long serialVersionUID = 1L;
  private int K = 2;
  private int maxIterations = 100;
  private boolean keepPoints = true;

  public DistributedKMeans() {
  }

  public DistributedKMeans(int K, int maxIterations) {
    this.K = K;
    this.maxIterations = maxIterations;
  }

  @Override
  public FlatClustering cluster(MStream<com.davidbracewell.apollo.linalg.Vector> instanceStream) {


    JavaRDD<Vector> rdd = new SparkStream<>(instanceStream)
      .map(v -> (Vector) new DenseVector(v.toArray()))
      .asRDD();


    KMeansModel model = org.apache.spark.mllib.clustering.KMeans.train(
      rdd.rdd(),
      K,
      maxIterations
    );

    Vector[] centroids = model.clusterCenters();
    List<Cluster> clusters = new ArrayList<>();
    for (int i = 0; i < centroids.length; i++) {
      clusters.add(null);
    }

    for (Iterator<Tuple2<Integer, Iterable<Vector>>> itr = model.predict(rdd).zip(rdd)
      .groupByKey()
      .toLocalIterator(); itr.hasNext(); ) {
      Tuple2<Integer, Iterable<Vector>> tuple = itr.next();
      Cluster cluster = new Cluster();
      int i = tuple._1();
      cluster.setCentroid(new com.davidbracewell.apollo.linalg.DenseVector(centroids[i].toArray()));
      if (isKeepPoints()) {
        tuple._2().forEach(lp -> {
          cluster.addPoint(new com.davidbracewell.apollo.linalg.DenseVector(lp.toArray()));
        });
      }
      clusters.set(i, cluster);
    }

    return new KMeansClustering(getEncoderPair(), Distance.Euclidean, clusters);
  }


  public boolean isKeepPoints() {
    return keepPoints;
  }

  public void setKeepPoints(boolean keepPoints) {
    this.keepPoints = keepPoints;
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
