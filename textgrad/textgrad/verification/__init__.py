from .general_purpose_verifier import GeneralPurposeVerifier

__all__ = [
    'GeneralPurposeVerifier',
]

# Integrate exploration to verify TextGrad with create new Loss Function, I want the verifier can used generic like verifier=GeneralPurposeVerifier