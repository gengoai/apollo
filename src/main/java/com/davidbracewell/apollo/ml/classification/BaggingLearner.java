package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.function.SerializableSupplier;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.ArrayList;

/**
 * The type Bagging learner.
 *
 * @author David B. Bracewell
 */
public class BaggingLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter(onParam = @_({@NonNull}))
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
    * @param learnerSupplier the learner supplier
    * @param numberOfBags    the number of bags
    * @param bagSize         the bag size
    */
   public BaggingLearner(@NonNull SerializableSupplier<ClassifierLearner> learnerSupplier, int numberOfBags, int bagSize) {
      this.learnerSupplier = learnerSupplier;
      this.numberOfBags = numberOfBags;
      this.bagSize = bagSize;
   }

   protected Classifier trainImpl(Dataset<Instance> dataset) {
      Ensemble model = new Ensemble(dataset.getEncoderPair(),
                                    dataset.getPreprocessors());
      dataset = dataset.shuffle();
      model.models = new ArrayList<>(numberOfBags);
      final int targetBagSize = (bagSize <= 0) ? dataset.size() : bagSize;
      for (int i = 0; i < numberOfBags; i++) {
         model.models.add(learnerSupplier.get().train(dataset.sample(true, targetBagSize)));
      }
      return model;
   }

   @Override
   public void reset() {

   }


}// END OF BaggingLearner
