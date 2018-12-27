package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.math.Math2;
import com.gengoai.stream.StreamingContext;
import com.gengoai.string.TableFormatter;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class SilhouetteEvaluation implements ClusteringEvaluation, Serializable {
   private static final long serialVersionUID = 1L;
   double avgSilhouette = 0;
   private Map<Integer, Double> silhouette;
   private final Measure measure;

   public SilhouetteEvaluation(Measure measure) {
      this.measure = measure;
   }

   public void evaluate(Clustering clustering) {
      Map<Integer, Cluster> idClusterMap = new HashMap<>();
      clustering.forEach(c -> idClusterMap.put(c.getId(), c));
      silhouette = StreamingContext.local()
                                   .stream(idClusterMap.keySet())
                                   .parallel()
                                   .mapToPair(i -> $(i, silhouette(idClusterMap, i, measure)))
                                   .collectAsMap();
      avgSilhouette = Math2.summaryStatistics(silhouette.values()).getAverage();

   }


   @Override
   public void merge(Evaluation evaluation) {
      throw new UnsupportedOperationException();
   }

   public double silhouette(Map<Integer, Cluster> clusters, int index, Measure distanceMeasure) {
      Cluster c1 = clusters.get(index);
      if (c1.size() <= 1) {
         return 0;
      }
      double s = 0;
      for (NDArray point1 : c1) {
         double ai = 0;
         for (NDArray point2 : c1) {
            ai += distanceMeasure.calculate(point1, point2);
         }
         ai = Double.isFinite(ai) ? ai : Double.MAX_VALUE;
         ai /= c1.size();
         double bi = clusters.keySet().parallelStream()
                             .filter(j -> j != index)
                             .mapToDouble(j -> {
                                if (clusters.get(j).size() == 0) {
                                   return Double.MAX_VALUE;
                                }
                                double b = 0;
                                for (NDArray point2 : clusters.get(j)) {
                                   b += distanceMeasure.calculate(point1, point2);
                                }
                                return b;
                             }).min()
                             .orElse(0);
         s += (bi - ai) / Math.max(bi, ai);
      }

      return s / c1.size();
   }

   public void reset() {
      this.avgSilhouette = 0;
      this.silhouette.clear();
   }

   @Override
   public void output(PrintStream printStream) {
      TableFormatter formatter = new TableFormatter();
      formatter.title("Silhouette Cluster Evaluation");
      formatter.header(Arrays.asList("Cluster", "Silhouette Score"));
      silhouette.keySet()
                .stream()
                .sorted()
                .forEach(id -> formatter.content(Arrays.asList(id, silhouette.get(id))));
      formatter.footer(Arrays.asList("Avg. Score", avgSilhouette));
      formatter.print(printStream);
   }

   public double getAvgSilhouette() {
      return avgSilhouette;
   }

   public double getSilhouette(int id) {
      return silhouette.get(id);
   }


}//END OF SilhouetteEvaluation
