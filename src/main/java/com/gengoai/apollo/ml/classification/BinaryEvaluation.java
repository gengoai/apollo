package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.encoder.LabelEncoder;
import com.gengoai.mango.conversion.Cast;
import com.gengoai.mango.string.TableFormatter;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.encoder.LabelEncoder;
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

      prob[goldClass].add(distribution[1]);
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
      prob[0].sort();
      prob[1].sort();

      int n0 = prob[0].size();
      int n1 = prob[1].size();

      int i0 = 0, i1 = 0;
      int rank = 1;
      double sum = 0d;

      while (i0 < n0 && i1 < n1) {
         double v0 = prob[0].get(i0);
         double v1 = prob[1].get(i1);

         if (v0 < v1) {
            i0++;
            rank++;
         } else if (v1 < v0) {
            i1++;
            sum += rank;
            rank++;
         } else {
            double tie = v0;

            int k0 = 0;
            while (i0 < n0 && prob[0].get(i0) == tie) {
               k0++;
               i0++;
            }


            int k1 = 0;
            while (i1 < n1 && prob[1].get(i1) == tie) {
               k1++;
               i1++;
            }


            sum += (rank + (k0 + k1 - 1) / 2.0) * k1;
            rank += k0 + k1;
         }
      }

      if (i1 < n1) {
         sum += (rank + (n1 - i1 - 1) / 2.0) * (n1 - i1);
         rank += (n1 - i1);
      }


      return (sum / n1 - (n1 + 1.0) / 2.0) / n0;
   }

   @Override
   public void output(PrintStream printStream) {
      TableFormatter tableFormatter = new TableFormatter();
      tableFormatter.header(Arrays.asList("Predicted / Gold", "TRUE", "FALSE", "TOTAL"));
      tableFormatter.content(
         Arrays.asList("TRUE", truePositives(), falsePositives(), (truePositives() + falsePositives())));
      tableFormatter.content(Arrays.asList("FALSE", falseNegatives(), trueNegatives(), (falseNegatives() +
                                                                                           trueNegatives())));
      tableFormatter.footer(
         Arrays.asList("", (truePositives() + falseNegatives()), (falsePositives() + trueNegatives()),
                       positive + negative));
      tableFormatter.print(printStream);

      tableFormatter = new TableFormatter();
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
