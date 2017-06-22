package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.function.SerializableSupplier;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.ArrayList;

/**
 * <p>Learner which takes random samples (with replacement) of the data to build a number of weaker models that each
 * vote in one ensemble model.</p>
 *
 * @author David B. Bracewell
 */
public class BaggingLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   private SerializableSupplier<ClassifierLearner> learnerSupplier;
   @Getter
   @Setter
   private int numberOfBags;
   @Getter
   @Setter
   private int bagSize;

   /**
    * Instantiates a new Bagging learner.
    */
   public BaggingLearner() {
      this.learnerSupplier = LibLinearLearner::new;
      this.numberOfBags = 10;
      this.bagSize = -1;
   }

   /**
    * Instantiates a new Bagging learner.
    *
    * @param learnerSupplier Supplier for the weak learner
    * @param numberOfBags    the number of bags, or weak learners, to generate
    * @param bagSize         the size of each random sample
    */
   public BaggingLearner(@NonNull SerializableSupplier<ClassifierLearner> learnerSupplier, int numberOfBags, int bagSize) {
      this.learnerSupplier = learnerSupplier;
      this.numberOfBags = numberOfBags;
      this.bagSize = bagSize;
   }

   @Override
   public void resetLearnerParameters() {

   }

   /**
    * Sets the supplier to use to generate weak learners
    *
    * @param learnerSupplier the learner supplier
    */
   public void setLearnerSupplier(@NonNull SerializableSupplier<ClassifierLearner> learnerSupplier) {
      this.learnerSupplier = learnerSupplier;
   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      Ensemble model = new Ensemble(this);
      dataset = dataset.shuffle();
      model.models = new ArrayList<>(numberOfBags);
      final int targetBagSize = (bagSize <= 0) ? dataset.size() : bagSize;
      for (int i = 0; i < numberOfBags; i++) {
         model.models.add(learnerSupplier.get().train(dataset.sample(true, targetBagSize)));
      }
      return model;
   }


}// END OF BaggingLearner
