package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.math.Math2;
import com.gengoai.stream.MStream;
import com.gengoai.string.TableFormatter;
import org.apache.mahout.math.list.DoubleArrayList;

import java.io.PrintStream;
import java.util.Arrays;

/**
 * <p>Common methods and metrics for evaluating a binary classifier.</p>
 *
 * @author David B. Bracewell
 */
public class BinaryEvaluation extends ClassifierEvaluation {
   private static final long serialVersionUID = 1L;
   private final String trueLabel;
   private final DoubleArrayList[] prob = {new DoubleArrayList(), new DoubleArrayList()};
   private double fn = 0;
   private double fp = 0;
   private double negative = 0d;
   private double positive = 0d;
   private double tn = 0;
   private double tp = 0;


   /**
    * Instantiates a new Binary evaluation.
    *
    * @param vectorizer the vectorizer
    */
   public BinaryEvaluation(Vectorizer<String> vectorizer) {
      this(vectorizer.decode(1.0));
   }

   public BinaryEvaluation(String trueLabel) {
      this.trueLabel = trueLabel;
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
      return Math2.auc(prob[0].elements(),
                       prob[1].elements());
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
      NDArray distribution = predicted.distribution();
      int predictedClass = predicted.getResult().equals(trueLabel) ? 1 : 0;
      int goldClass = gold.equals(trueLabel) ? 1 : 0;

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
   public void evaluate(Classifier model, MStream<Example> dataset) {
      dataset.forEach(instance -> entry(instance.getLabel(), model.predict(instance)));
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

   private double toDouble(String string) {
      return string.equals(trueLabel) ? 1.0 : 0.0;
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
