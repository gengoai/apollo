package com.davidbracewell.apollo.ml;

import lombok.NonNull;
import lombok.Value;

import java.io.Serializable;

/**
 * The type Train test.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
@Value
public class TrainTest<T extends Example> implements Serializable {
  private Dataset<T> train;
  private Dataset<T> test;

  /**
   * Of train test.
   *
   * @param <T>   the type parameter
   * @param train the train
   * @param test  the test
   * @return the train test
   */
  public static <T extends Example> TrainTest<T> of(@NonNull Dataset<T> train, @NonNull Dataset<T> test) {
    return new TrainTest<>(train, test);
  }

}// END OF TrainTest
