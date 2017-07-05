package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.optimization.WeightInitializer;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public abstract class LayerBuilder implements Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int inputSize;
   @Getter
   @Setter
   private int outputSize = 100;
   @Getter
   @Setter
   private WeightInitializer weightInitializer = WeightInitializer.DEFAULT;

   public abstract Layer build();

}// END OF LayerBuilder
