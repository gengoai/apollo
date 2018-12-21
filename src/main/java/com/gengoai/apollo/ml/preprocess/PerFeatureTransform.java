package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.function.SerializableFunction;
import com.gengoai.string.Strings;

import java.io.Serializable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author David B. Bracewell
 */
public class PerFeatureTransform implements Preprocessor, Serializable {
   private static final long serialVersionUID = 1L;
   private final PreprocessorList preprocessors = new PreprocessorList();
   private final SerializableFunction<String, Preprocessor> preprocessorSupplier;

   public PerFeatureTransform(SerializableFunction<String, Preprocessor> preprocessorSupplier) {
      this.preprocessorSupplier = preprocessorSupplier;
   }

   @Override
   public Example apply(Example example) {
      return preprocessors.apply(example);
   }

   @Override
   public Dataset fitAndTransform(Dataset dataset) {
      final Pattern prefixPattern = Pattern.compile("^(.+?)=(.+?)$");
      IndexVectorizer features = IndexVectorizer.featureVectorizer();
      features.fit(dataset);
      features.alphabet()
              .stream()
              .map(f -> {
                 Matcher m = prefixPattern.matcher(f);
                 if (m.find()) {
                    return m.group(1);
                 }
                 return f;
              }).distinct()
              .filter(Strings::isNotNullOrBlank)
              .map(preprocessorSupplier)
              .forEach(preprocessors::add);
      return preprocessors.fitAndTransform(dataset);
   }

   @Override
   public void reset() {
      preprocessors.clear();
   }


}//END OF PerFeatureTransform
