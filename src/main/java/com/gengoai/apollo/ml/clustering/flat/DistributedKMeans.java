package com.gengoai.apollo.ml.clustering.flat;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.mango.collection.list.Lists;
import com.gengoai.mango.stream.MStream;
import com.gengoai.mango.stream.SparkStream;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import lombok.Getter;
import lombok.Setter;
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
 * Wrapper around Spark's implementation of K-Means.
 *
 * @author David B. Bracewell
 */
public class DistributedKMeans extends Clusterer<FlatClustering> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int K = 2;
   @Getter
   @Setter
   private int maxIterations = 100;
   @Getter
   @Setter
   private boolean keepPoints = true;

   /**
    * Instantiates a new Distributed K-means.
    */
   public DistributedKMeans() {
   }

   /**
    * Instantiates a new Distributed K-means.
    *
    * @param K             the number of clusters
    * @param maxIterations the maximum number of iterations to run the algorithm
    */
   public DistributedKMeans(int K, int maxIterations) {
      this.K = K;
      this.maxIterations = maxIterations;
   }

   @Override
   public FlatClustering cluster(MStream<NDArray> instanceStream) {
      SparkStream<NDArray> sparkStream = new SparkStream<>(instanceStream);
      JavaRDD<Vector> rdd = sparkStream.getRDD().map(v -> (Vector) new DenseVector(v.toArray())).cache();
      JavaRDD<String> labels = sparkStream.getRDD()
                                          .map(o -> {
                                             if (o.getLabel() instanceof Number) {
                                                return getEncoderPair().decodeLabel(o.getLabel()).toString();
                                             }
                                             return o.getLabel().toString();
                                          })
                                          .cache();

      JavaPairRDD<String, Vector> pair = labels.zip(rdd);

      KMeansModel model = org.apache.spark.mllib.clustering.KMeans.train(rdd.rdd(), K, maxIterations);

      Vector[] centroids = model.clusterCenters();
      List<Cluster> clusters = new ArrayList<>();
      Lists.ensureSize(clusters, K, null);

      for (Iterator<Tuple2<Integer, Iterable<Tuple2<String, Vector>>>> itr = model.predict(rdd).zip(pair)
                                                                                  .groupByKey()
                                                                                  .toLocalIterator(); itr.hasNext(); ) {
         Tuple2<Integer, Iterable<Tuple2<String, Vector>>> tuple = itr.next();
         Cluster cluster = new Cluster();
         int i = tuple._1();
         cluster.setCentroid(NDArrayFactory.wrap(centroids[i].toArray()));
         if (isKeepPoints()) {
            tuple._2().forEach(lp -> cluster.addPoint(
               NDArrayFactory.wrap(lp._2().toArray()).setLabel(lp._1())));
         }
         clusters.set(i, cluster);
      }

      return new FlatCentroidClustering(this, Distance.Euclidean, clusters);
   }

}// END OF DistributedKMeans
