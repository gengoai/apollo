package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.apollo.stat.measure.Distance;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.concurrent.AtomicDouble;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import com.gengoai.tuple.Tuple2;
import org.apache.mahout.math.map.OpenIntIntHashMap;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import static com.gengoai.tuple.Tuples.$;

/**
 * The type K medoids.
 *
 * @author David B. Bracewell
 */
public class KMedoids extends FlatClusterer {
   private static final long serialVersionUID = 1L;

   public KMedoids() {
   }

   public KMedoids(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public KMedoids(Vectorizer<String> featureVectorizer, PreprocessorList preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   public KMedoids(Vectorizer<?> labelVectorizer, Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(labelVectorizer, featureVectorizer, preprocessors);
   }

   public KMedoids(Vectorizer<String> featureVectorizer, Preprocessor... preprocessors) {
      super(featureVectorizer, preprocessors);
   }

   private double distance(int i, int j, List<NDArray> instances, Map<Tuple2<Integer, Integer>, Double> distances) {
      return distances.computeIfAbsent($(i, j), t -> measure.calculate(instances.get(i), instances.get(j)));
   }

   /**
    * Fit.
    *
    * @param dataSupplier  the data supplier
    * @param fitParameters the fit parameters
    */
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      this.measure = fitParameters.measure;
      final List<NDArray> instances = dataSupplier.get().collect();
      Map<Tuple2<Integer, Integer>, Double> distanceCache = new ConcurrentHashMap<>();

      List<TempCluster> tempClusters = new ArrayList<>();
      OpenIntHashSet seen = new OpenIntHashSet();
      while (seen.size() < fitParameters.K) {
         seen.add((int) Math.round(Math.random() * fitParameters.K));
      }
      seen.forEachKey(i -> tempClusters.add(new TempCluster().centroid(i)));

      OpenIntIntHashMap assignments = new OpenIntIntHashMap();

      for (int iteration = 0; iteration < fitParameters.maxIterations; iteration++) {
         System.err.println("iteration " + iteration);
         AtomicLong numChanged = new AtomicLong();
         tempClusters.forEach(c -> c.points.clear());
         IntStream.range(0, instances.size())
                  .forEach((int i) -> {
                     double minDistance = Double.POSITIVE_INFINITY;
                     int minC = -1;
                     for (int c = 0; c < tempClusters.size(); c++) {
                        TempCluster cluster = tempClusters.get(c);
                        if (cluster.centroid == i) {
                           minC = c;
                           break;
                        }
                        double d = distance(i, cluster.centroid, instances, distanceCache);
                        if (d < minDistance) {
                           minC = c;
                           minDistance = d;
                        }
                     }
                     int old = assignments.containsKey(i) ? assignments.get(i) : -1;
                     assignments.put(i, minC);
                     if (old != minC) {
                        numChanged.incrementAndGet();
                        tempClusters.get(minC).points.add(i);
                     }
                  });

         if (numChanged.get() == 0) {
            break;
         }


         tempClusters
                     .forEach(c -> {
                        AtomicInteger minPoint = new AtomicInteger();
                        AtomicDouble minDistance = new AtomicDouble(Double.POSITIVE_INFINITY);
                        c.points.forEachKey(i -> {
                           AtomicDouble sum = new AtomicDouble();
                           AtomicLong total = new AtomicLong();
                           c.points.forEachKey(j -> {
                              total.incrementAndGet();
                              sum.addAndGet(distance(i, j, instances, distanceCache));
                              return true;
                           });
                           double avg = sum.get() / total.get();
                           if (avg < minDistance.get()) {
                              minDistance.set(avg);
                              minPoint.set(i);
                           }
                           return true;
                        });
                        c.centroid(minPoint.get());
                     });
      }

      clusters.clear();
      AtomicInteger cid = new AtomicInteger();
      tempClusters.forEach(tc -> {
         Cluster c = new Cluster();
         c.setId(cid.getAndIncrement());
         c.setCentroid(instances.get(tc.centroid));
         AtomicDouble sum = new AtomicDouble();
         AtomicLong total = new AtomicLong();
         tc.points.forEachKey(i -> {
            c.addPoint(instances.get(i));
            tc.points.forEachKey(j -> {
               total.incrementAndGet();
               sum.addAndGet(distance(i, j, instances, distanceCache));
               return true;
            });
            return true;
         });
         c.setScore(sum.get() / total.get());
         clusters.add(c);
      });

   }

   @Override
   public void fitPreprocessed(Dataset dataSupplier, FitParameters fitParameters) {
      fit(() -> dataSupplier.stream().map(this::encode), Cast.as(fitParameters, Parameters.class));
   }


   @Override
   public FitParameters getDefaultFitParameters() {
      return new Parameters();
   }

   /**
    * The type Parameters.
    */
   public static class Parameters extends FitParameters {
      /**
       * The K.
       */
      public int K = 2;
      /**
       * The Max iterations.
       */
      public int maxIterations = 100;
      /**
       * The Measure.
       */
      public Measure measure = Distance.Euclidean;
   }

   private static class TempCluster {
      /**
       * The Centroid.
       */
      int centroid;
      /**
       * The Points.
       */
      OpenIntHashSet points = new OpenIntHashSet();


      /**
       * Centroid int.
       *
       * @return the int
       */
      int centroid() {
         return centroid;
      }

      /**
       * Centroid temp cluster.
       *
       * @param centroid the centroid
       * @return the temp cluster
       */
      TempCluster centroid(int centroid) {
         this.centroid = centroid;
         return this;
      }
   }

}//END OF KMedoids
