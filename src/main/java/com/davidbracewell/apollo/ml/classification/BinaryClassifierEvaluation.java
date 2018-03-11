package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.encoder.LabelEncoder;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.tuple.Tuple2;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class BinaryClassifierEvaluation implements Evaluation<Instance, Classifier>, Serializable {

   private final List<Tuple2<Boolean, Double>> results = new ArrayList<>();
   private double positive = 0d;
   private double negative = 0d;
   private final String positiveLabel;

   public BinaryClassifierEvaluation(LabelEncoder labelEncoder) {
      this.positiveLabel = labelEncoder.decode(1.0).toString();
   }


   public void entry(String gold, double[] distribution) {
      results.add($(gold.equals(positiveLabel), distribution[1]));
      if (gold.equals(positiveLabel)) {
         positive++;
      } else {
         negative++;
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
      if (evaluation instanceof BinaryClassifierEvaluation) {
         BinaryClassifierEvaluation bce = Cast.as(evaluation);
         this.results.addAll(bce.results);
      } else {
         throw new IllegalArgumentException();
      }
   }

   public double auc() {
      results.sort(Comparator.comparing(Tuple2::getV2));
      double[] rank = new double[results.size()];
      for (int i = 0; i < results.size(); i++) {
         double conf = results.get(i).v2;

         if (i + 1 == results.size() || conf != results.get(i + 1).v2) {
            rank[i] = i + 1;
         } else {
            int j = i + 1;
            for (; j < results.size() && conf == results.get(j).v2; j++) ;
            double r = (i + 1 + j) / 2.0;
            for (int k = i; k < j; k++) rank[k] = r;
            i = j - 1;
         }
      }

      double auc = 0;
      for (int i = 0; i < results.size(); i++) {
         if (results.get(i).v1) {
            auc += rank[i];
         }
      }

      auc = (auc - (positive * (positive + 1) / 2.0)) / (positive * negative);
      return auc;
   }

   @Override
   public void output(PrintStream printStream) {

   }


}//END OF BinaryClassifierEvaluation
