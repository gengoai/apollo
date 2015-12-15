package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.classification.OneVsRestLearner;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.reflection.BeanMap;
import com.davidbracewell.reflection.Reflect;
import com.davidbracewell.reflection.ReflectionException;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import lombok.NonNull;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public class LearnerBuilder<T extends Example, M extends Model> implements Serializable {
  private static final long serialVersionUID = 1L;
  @Setter(onParam = @_({@NonNull}))
  private Class<? extends Learner<T, M>> learnerClass;
  @Setter(onParam = @_({@NonNull}))
  private Map<String, Object> parameters = new HashMap<>();


  public LearnerBuilder<T, M> parameter(String name, Object value) {
    parameters.put(name, value);
    return this;
  }

  public ClassifierLearner oneVsRest() {
    return new OneVsRestLearner(() -> Cast.as(this.build()));
  }

  public <R extends Learner<T, M>> R build() {
    Preconditions.checkNotNull(learnerClass, "Learner was not set");
    try {
      BeanMap beanMap = new BeanMap(Reflect.onClass(learnerClass).create().get());
      beanMap.putAll(parameters);
      return Cast.as(beanMap.getBean());
    } catch (ReflectionException e) {
      throw Throwables.propagate(e);
    }
  }


}// END OF LearnerBuilder
