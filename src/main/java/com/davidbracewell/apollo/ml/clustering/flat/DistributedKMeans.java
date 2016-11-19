package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.collection.list.Lists;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.SparkStream;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * The type Distributed k means.
 *
 * @author David B. Bracewell
 */
public class DistributedKMeans extends Clusterer<FlatClustering> {
   private static final long serialVersionUID = 1L;
   private int K = 2;
   private int maxIterations = 100;
   private boolean keepPoints = true;

   /**
    * Instantiates a new Distributed k means.
    */
   public DistributedKMeans() {
   }

   /**
    * Instantiates a new Distributed k means.
    *
    * @param K             the k
    * @param maxIterations the max iterations
    */
   public DistributedKMeans(int K, int maxIterations) {
      this.K = K;
      this.maxIterations = maxIterations;
   }

   @Override
   public FlatClustering cluster(MStream<com.davidbracewell.apollo.linalg.Vector> instanceStream) {
      SparkStream<com.davidbracewell.apollo.linalg.Vector> sparkStream = new SparkStream<>(instanceStream);
      JavaRDD<Vector> rdd = sparkStream.getRDD().map(v -> (Vector) new DenseVector(v.toArray())).cache();
      JavaRDD<String> labels = sparkStream.getRDD()
                                          .map(o -> getEncoderPair().decodeLabel(o.getLabel()).toString())
                                          .cache();

      JavaPairRDD<String, Vector> pair = labels.zip(rdd);

      KMeansModel model = org.apache.spark.mllib.clustering.KMeans.train(rdd.rdd(),
                                                                         K,
                                                                         maxIterations
                                                                        );

      Vector[] centroids = model.clusterCenters();
      List<Cluster> clusters = new ArrayList<>();
      Lists.ensureSize(clusters, K, null);

      for (Iterator<Tuple2<Integer, Iterable<Tuple2<String, Vector>>>> itr = model.predict(rdd).zip(pair)
                                                                                  .groupByKey()
                                                                                  .toLocalIterator(); itr.hasNext(); ) {
         Tuple2<Integer, Iterable<Tuple2<String, Vector>>> tuple = itr.next();
         Cluster cluster = new Cluster();
         int i = tuple._1();
         cluster.setCentroid(new com.davidbracewell.apollo.linalg.DenseVector(centroids[i].toArray()));
         if (isKeepPoints()) {
            tuple._2().forEach(lp -> {
               LabeledVector dv = new LabeledVector(lp._1(),
                                                    new com.davidbracewell.apollo.linalg.DenseVector(lp._2().toArray())
               );
               cluster.addPoint(dv);
            });
         }
         clusters.set(i, cluster);
      }

      return new FlatCentroidClustering(getEncoderPair(), Distance.Euclidean, clusters);
   }


   /**
    * Is keep points boolean.
    *
    * @return the boolean
    */
   public boolean isKeepPoints() {
      return keepPoints;
   }

   /**
    * Sets keep points.
    *
    * @param keepPoints the keep points
    */
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
