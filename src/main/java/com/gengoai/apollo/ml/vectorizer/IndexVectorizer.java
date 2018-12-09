package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Dataset;
import com.gengoai.collection.HashMapIndex;
import com.gengoai.collection.Index;
import com.gengoai.collection.Streams;
import com.gengoai.json.JsonEntry;
import com.gengoai.reflection.Types;

import java.lang.reflect.Type;

/**
 * @author David B. Bracewell
 */
public class IndexVectorizer extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   private final Index<String> index = new HashMapIndex<>();
   private final boolean isLabel;
   private final String unknown;

   public IndexVectorizer(boolean isLabel) {
      this(isLabel, null);
   }


   public IndexVectorizer(boolean isLabel, String unknown) {
      this.isLabel = isLabel;
      this.unknown = unknown;
      if (this.unknown != null) {
         index.add(this.unknown);
      }
   }

   public static IndexVectorizer featureVectorizer() {
      return new IndexVectorizer(false);
   }

   public static IndexVectorizer featureVectorizer(String unknownFeature) {
      return new IndexVectorizer(false, unknownFeature);
   }

   public static IndexVectorizer fromJson(JsonEntry entry, Type... parameters) {
      Validation.checkState(entry.getStringProperty("class").equalsIgnoreCase(IndexVectorizer.class.getName()));
      IndexVectorizer vectorizer = new IndexVectorizer(entry.getBooleanProperty("isLabel"));
      vectorizer.index.addAll(entry.getProperty("index").getAs(Types.parameterizedType(Index.class, String.class)));
      return vectorizer;
   }

   public static IndexVectorizer labelVectorizer() {
      return new IndexVectorizer(true);
   }

   @Override
   public String decode(double value) {
      return index.get((int) value);
   }

   @Override
   public double encode(String value) {
      int i = index.getId(value);
      if (i < 0 && unknown != null) {
         return index.getId(unknown);
      }
      return i;
   }

   @Override
   public void fit(Dataset dataset) {
      index.clear();
      index.addAll(dataset.stream()
                          .flatMap(Streams::asStream)
                          .flatMap(example -> {
                             if (isLabel) {
                                return example.getStringLabelSpace();
                             }
                             return example.getFeatureNameSpace();
                          })
                          .distinct()
                          .collect());
   }

   @Override
   public boolean isLabelVectorizer() {
      return isLabel;
   }

   @Override
   public int size() {
      return index.size();
   }

   public JsonEntry toJson() {
      return JsonEntry.object()
                      .addProperty("class", IndexVectorizer.class)
                      .addProperty("isLabel", isLabel)
                      .addProperty("index", index);
   }

   @Override
   public String toString() {
      return "IndexVectorizer{" +
                "size=" + index.size() +
                ", vectorizing=" + (isLabel ? "Label" : "Feature") +
                '}';
   }
}//END OF IndexEncoder
