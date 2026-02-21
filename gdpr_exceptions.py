"""
GDPR Consent Exceptions

Определяет исключения для обработки сценариев, связанных с согласием пользователя
на обработку персональных данных в соответствии с GDPR.
"""


class ConsentRequiredError(Exception):
    """
    Raised when operation requires user consent but consent was not given
    
    This exception provides explicit handling for GDPR compliance:
    - Caller can distinguish "no consent" from other errors
    - Type system correctly reflects that successful operations return int, not Optional[int]
    - Enables user-friendly error messages explaining consent requirement
    
    Usage in sync code:
        try:
            query_id = db.log_query(user_id, query_text)
        except ConsentRequiredError as e:
            logger.warning(f"User {e.user_id} attempted operation without consent")
            # Show consent request to user
    
    Usage in async code:
        try:
            query_id = await save_user_query(user_id, query_text, answer_text, query_type)
        except ConsentRequiredError as e:
            await message.answer(
                "⚠️ Для работы бота необходимо согласие на обработку данных.\n\n"
                "Используйте /start для принятия условий."
            )
    
    Attributes:
        user_id: ID пользователя, который пытался выполнить операцию без согласия
    """
    def __init__(self, message: str, user_id: int = None):
        self.user_id = user_id
        super().__init__(message)
    
    def __str__(self):
        if self.user_id:
            return f"Consent required for user {self.user_id}: {super().__str__()}"
        return super().__str__()


class ConsentAlreadyGivenError(Exception):
    """
    Raised when trying to grant consent but user already gave consent
    
    Это информационное исключение для предотвращения дублирования согласия.
    Не является ошибкой в строгом смысле, но позволяет caller знать,
    что согласие уже было предоставлено.
    """
    def __init__(self, user_id: int):
        self.user_id = user_id
        super().__init__(f"User {user_id} already gave consent")
