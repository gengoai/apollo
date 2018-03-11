package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.encoder.LabelEncoder;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.string.TableFormatter;
import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.mahout.math.list.DoubleArrayList;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;

/**
 * @author David B. Bracewell
 */
public class BinaryEvaluation implements ClassifierEvaluation {
   private static final long serialVersionUID = 1L;

   private final DoubleArrayList[] prob = {new DoubleArrayList(), new DoubleArrayList()};
   private double positive = 0d;
   private double negative = 0d;
   private double tp = 0;
   private double tn = 0;
   private double fp = 0;
   private double fn = 0;
   private final String positiveLabel;

   public BinaryEvaluation(LabelEncoder labelEncoder) {
      this.positiveLabel = labelEncoder.decode(1.0).toString();
   }

   @Override
   public double accuracy() {
      return (tp + tn) / (positive + negative);
   }

   @Override
   public double falseNegatives() {
      return fn;
   }

   @Override
   public double falsePositives() {
      return fp;
   }

   @Override
   public double truePositives() {
      return tp;
   }

   @Override
   public double trueNegatives() {
      return tn;
   }


   public void entry(String gold, double[] distribution) {
      int predictedClass = distribution[1] >= 0.5 ? 1 : 0;
      int goldClass = gold.equals(positiveLabel) ? 1 : 0;

      prob[predictedClass].add(distribution[1]);
      if (goldClass == 1) {
         positive++;
         if (predictedClass == 1) {
            tp++;
         } else {
            fn++;
         }
      } else {
         negative++;
         if (predictedClass == 1) {
            fp++;
         } else {
            tn++;
         }
      }
   }


   @Override
   public void evaluate(Classifier model, Dataset<Instance> dataset) {
      dataset.forEach(instance -> {
         Classification clf = model.classify(instance);
         double dist[] = clf.distribution();
         entry(instance.getLabel().toString(), dist);
      });
   }

   @Override
   public void evaluate(Classifier model, Collection<Instance> dataset) {
      dataset.forEach(instance -> {
         Classification clf = model.classify(instance);
         double dist[] = clf.distribution();
         entry(instance.getLabel().toString(), dist);
      });
   }

   @Override
   public void merge(Evaluation<Instance, Classifier> evaluation) {
      if (evaluation instanceof BinaryEvaluation) {
         BinaryEvaluation bce = Cast.as(evaluation);
         this.prob[0].addAllOf(bce.prob[0]);
         this.prob[1].addAllOf(bce.prob[1]);
      } else {
         throw new IllegalArgumentException();
      }
   }

   public double auc() {
      MannWhitneyUTest mwu = new MannWhitneyUTest();
      return mwu.mannWhitneyU(prob[0].elements(), prob[1].elements());
//      double[] x = new double[results.size()];
//      double[] y = new double[results.size()];
//      for (int i = 0; i < results.size(); i++) {
//         y[i] = results.get(i).v1 ? 1.0 : 0.0;
//         x[i] = results.get(i).v2;
//      }
//      double auc = mwu.mannWhitneyU(x, y);
//      results.sort(Comparator.comparing(Tuple2::getV2));
//      double[] rank = new double[results.size()];
//      for (int i = 0; i < results.size(); i++) {
//         double conf = results.get(i).v2;
//
//         if (i + 1 == results.size() || conf != results.get(i + 1).v2) {
//            rank[i] = i + 1;
//         } else {
//            int j = i + 1;
//            for (; j < results.size() && conf == results.get(j).v2; j++) ;
//            double r = (i + 1 + j) / 2.0;
//            for (int k = i; k < j; k++) rank[k] = r;
//            i = j - 1;
//         }
//      }
//
//      double auc = 0;
//      for (int i = 0; i < results.size(); i++) {
//         if (results.get(i).v1) {
//            auc += rank[i];
//         }
//      }
//
//      auc = (auc - (positive * (positive + 1) / 2.0)) / (positive * negative);
//      return auc;
   }

   @Override
   public void output(PrintStream printStream) {
      TableFormatter tableFormatter = new TableFormatter();
      tableFormatter.header(Arrays.asList("Metric", "Score"));
      tableFormatter.content(Arrays.asList("AUC", auc()));
      tableFormatter.content(Arrays.asList("Accuracy", accuracy()));
      tableFormatter.content(Arrays.asList("TP Rate", truePositiveRate()));
      tableFormatter.content(Arrays.asList("FP Rate", falsePositiveRate()));
      tableFormatter.content(Arrays.asList("TN Rate", trueNegativeRate()));
      tableFormatter.content(Arrays.asList("FN Rate", falseNegativeRate()));
      tableFormatter.print(printStream);
   }


}//END OF BinaryClassifierEvaluation
