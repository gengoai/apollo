package com.davidbracewell.apollo.ml.sequence;

/**
 * @author David B. Bracewell
 */
public interface Decoder {

  LabelingResult decode(SequenceLabeler labeler, Sequence sequence);

}//END OF Decoder
