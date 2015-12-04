package com.davidbracewell.apollo.ml;

import lombok.NonNull;

import java.util.ArrayList;
import java.util.function.BiConsumer;

/**
 * @author David B. Bracewell
 */
public class TrainTestSet<T extends Example> extends ArrayList<TrainTest<T>> {

  public void forEach(@NonNull BiConsumer<Dataset<T>, Dataset<T>> consumer) {
    forEach(tTrainTest -> consumer.accept(tTrainTest.getTrain(), tTrainTest.getTest()));
  }

}// END OF TrainTestSet
