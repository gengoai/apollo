package com.gengoai.apollo.ml;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;

import java.io.Serializable;
import java.util.function.Consumer;

/**
 * <p>A generic interface to classification, regression, and sequence models. Models are trained using the
 * <code>fit</code> method which takes in the training data and {@link Parameters}. Models are used on new data by
 * calling the <code>transform</code> method. </p>
 *
 * @author David B. Bracewell
 */
public interface Model extends Serializable {

   /**
    * Reads a Classifier from the given resource
    *
    * @param resource the resource containing the saved model
    * @return the deserialized (loaded) model
    * @throws Exception Something went wrong reading in the model
    */
   static Model read(Resource resource) throws Exception {
      return resource.readObject();
   }

   /**
    * Estimates the outcome for a single NDArray. The return value is the given NDArray data with the estimation on the
    * <code>predicted</code> value.
    *
    * @param data the NDArray to transform.
    * @return the input data with the estimation on the <code>predicted</code> label ( {@link NDArray#getPredicted()} )
    */
   NDArray estimate(NDArray data);

   /**
    * Fits the model (i.e. trains) it on the given data using the given model parameters..
    *
    * @param dataSupplier  A supplier of {@link NDArray} representing training data
    * @param fitParameters the fit parameters
    */
   void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters fitParameters);

   /**
    * Fit.
    *
    * @param dataSupplier  the data supplier
    * @param fitParameters the fit parameters
    */
   default void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Consumer<? extends FitParameters> fitParameters) {
      FitParameters f = getDefaultFitParameters();
      fitParameters.accept(Cast.as(f));
      this.fit(dataSupplier, f);
   }

   /**
    * Gets default fit parameters.
    *
    * @return the default fit parameters
    */
   FitParameters getDefaultFitParameters();

   /**
    * Writes the model to the given resource.
    *
    * @param resource the resource to write the model to
    * @throws Exception Something went wrong writing the model
    */
   default void write(Resource resource) throws Exception {
      resource.setIsCompressed(true).writeObject(this);
   }

}//END OF Model
