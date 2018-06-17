package com.gengoai.apollo.ml.data.source;

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.collection.Iterators;
import com.gengoai.io.CSV;
import com.gengoai.io.CSVReader;
import com.gengoai.io.QuietIO;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.string.StringUtils;
import lombok.Getter;
import lombok.NonNull;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>A sparse CSV data source where each features are represented using <code>Feature:value</code>.</p>
 *
 * @author David B. Bracewell
 */
public class SparseCSVDataSource extends DataSource<Instance> {
   private static final long serialVersionUID = 1L;
   private final CSV csvFormat;
   @Getter
   private final String labelName;

   /**
    * Instantiates a new Sparse CSV data source. Assumes that there is no header in the file.
    *
    * @param resource  the resource containing the data
    * @param labelName the label name
    */
   public SparseCSVDataSource(@NonNull Resource resource, String labelName) {
      this(resource, labelName, CSV.builder());
   }


   /**
    * Instantiates a new Sparse CSV data source
    *
    * @param resource  the resource containing the data
    * @param labelName the label name
    * @param format    the CSV format to use for parsing the data
    */
   public SparseCSVDataSource(@NonNull Resource resource, String labelName, @NonNull CSV format) {
      super(resource);
      Validation.checkArgument(StringUtils.isNotNullOrBlank(labelName), "Must specify a label name.");
      this.csvFormat = format;
      this.labelName = labelName;
   }

   private MStream<Instance> resourceToStream(Resource resource, @NonNull StreamingContext context) throws IOException {
      CSVReader reader = csvFormat.reader(resource);
      MStream<Instance> stream = context.stream(Iterators.transform(reader.iterator(), list -> {
         List<Feature> features = new ArrayList<>();
         String label = null;
         for (String aList : list) {
            List<String> parts = StringUtils.split(aList, ':');
            if (parts.get(0).equals(labelName)) {
               label = parts.get(1);
            } else {
               if (parts.size() == 2) {
                  features.add(Feature.real(parts.get(0), Double.parseDouble(parts.get(1))));
               } else {
                  features.add(Feature.TRUE(parts.get(0)));
               }
            }
         }
         if (label == null) {
            return new Instance(features);
         }
         return new Instance(features, label);
      }));
      stream.onClose(() -> QuietIO.closeQuietly(reader));
      return stream;
   }

   @Override
   public MStream<Instance> stream() throws IOException {
      if (getResource().isDirectory()) {
         MStream<Instance> stream = getStreamingContext().empty();
         for (MStream<Instance> s : getResource().getChildren(true)
                                                 .stream()
                                                 .map(r -> {
                                                    try {
                                                       return resourceToStream(r, getStreamingContext());
                                                    } catch (IOException e) {
                                                       throw new RuntimeException(e);
                                                    }
                                                 }).collect(Collectors.toList())) {
            stream = stream.union(s);
         }
      }
      return resourceToStream(getResource(), getStreamingContext());
   }
}// END OF SparseCSVDataFormat
