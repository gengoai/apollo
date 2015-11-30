package com.davidbracewell.apollo.learning.classification;

import com.davidbracewell.apollo.learning.Instance;
import com.davidbracewell.stream.MStream;

import java.util.function.Supplier;

/**
 * @author David B. Bracewell
 */
public interface OnlineClassifierTrainer<T> {

  /**
   * Train classifier.
   *
   * @param instanceSupplier the instance supplier
   * @return the classifier
   */
  Classifier<T> train(Supplier<MStream<Instance>> instanceSupplier);


}//END OF OnlineClassifierTrainer
