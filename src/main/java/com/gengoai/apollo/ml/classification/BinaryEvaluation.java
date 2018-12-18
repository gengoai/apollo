package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.vectorizer.StringVectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.string.TableFormatter;
import org.apache.mahout.math.list.DoubleArrayList;

import java.io.PrintStream;
import java.util.Arrays;

/**
 * <p>Common methods and metrics for evaluating a binary classifier.</p>
 *
 * @author David B. Bracewell
 */
public class BinaryEvaluation implements ClassifierEvaluation {
   private static final long serialVersionUID = 1L;
   private final String goldLabel;
   private final DoubleArrayList[] prob = {new DoubleArrayList(), new DoubleArrayList()};
   private double fn = 0;
   private double fp = 0;
   private double negative = 0d;
   private double positive = 0d;
   private double tn = 0;
   private double tp = 0;

   /**
    * Instantiates a new Binary evaluation.
    */
   public BinaryEvaluation() {
      this.goldLabel = null;
   }

   /**
    * Instantiates a new Binary evaluation.
    *
    * @param goldLabel the gold label to use when comparing classification results
    */
   public BinaryEvaluation(String goldLabel) {
      this.goldLabel = goldLabel == null ? "true" : goldLabel;
   }


   /**
    * Instantiates a new Binary evaluation.
    *
    * @param vectorizer the vectorizer
    */
   public BinaryEvaluation(StringVectorizer vectorizer) {
      this(vectorizer.decode(1.0));
   }

   @Override
   public double accuracy() {
      return (tp + tn) / (positive + negative);
   }

   /**
    * Calculates the AUC (Area Under the Curve)
    *
    * @return the AUC
    */
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

   /**
    * Calculates the baseline score, which is <code>max(positive,negative) / (positive+negative)</code>
    *
    * @return the baseline score
    */
   public double baseline() {
      return Math.max(positive, negative) / (positive + negative);
   }

   @Override
   public void entry(String gold, Classification predicted) {
      entry(gold, predicted.distribution());
   }

   @Override
   public void entry(double gold, double predicted) {
      entry(gold == 1.0, NDArrayFactory.DENSE.scalar(predicted));
   }

   /**
    * Adds an entry into the metric.
    *
    * @param gold         the gold label
    * @param distribution the distribution of results (examines the positive class at index 1.0)
    */
   public void entry(String gold, NDArray distribution) {
      entry(gold.equals(goldLabel), distribution);
   }

   /**
    * Adds an entry into the metric.
    *
    * @param gold         the gold label (true or false)
    * @param distribution the distribution of results (examines the positive class at index 1.0)
    */
   public void entry(boolean gold, NDArray distribution) {
      int predictedClass = distribution.get(1) >= 0.5 ? 1 : 0;
      int goldClass = gold ? 1 : 0;

      prob[goldClass].add(distribution.get(1));
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
   public void entry(NDArray entry) {
      entry(entry.getLabelAsDouble() == 1.0, entry.getPredictedAsNDArray());
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
   public void merge(Evaluation evaluation) {
      if (evaluation instanceof BinaryEvaluation) {
         BinaryEvaluation bce = Cast.as(evaluation);
         this.prob[0].addAllOf(bce.prob[0]);
         this.prob[1].addAllOf(bce.prob[1]);
      } else {
         throw new IllegalArgumentException();
      }
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
      tableFormatter.content(Arrays.asList("Baseline", baseline()));
      tableFormatter.content(Arrays.asList("TP Rate", truePositiveRate()));
      tableFormatter.content(Arrays.asList("FP Rate", falsePositiveRate()));
      tableFormatter.content(Arrays.asList("TN Rate", trueNegativeRate()));
      tableFormatter.content(Arrays.asList("FN Rate", falseNegativeRate()));
      tableFormatter.print(printStream);
   }

   @Override
   public double trueNegatives() {
      return tn;
   }

   @Override
   public double truePositives() {
      return tp;
   }


}//END OF BinaryClassifierEvaluation
